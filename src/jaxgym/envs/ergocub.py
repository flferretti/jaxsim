import dataclasses
import functools
import multiprocessing
import os
import warnings
from typing import Any, Dict, List, Optional, Type, Union

import gymnasium as gym
import jax.numpy as jnp
import jax.random
import jax_dataclasses
import numpy as np
import numpy.typing as npt
import rod
from gymnasium.experimental.vector.vector_env import VectorWrapper
from meshcat_viz import MeshcatWorld
from resolve_robotics_uri_py import resolve_robotics_uri
from stable_baselines3 import PPO
from stable_baselines3.common import vec_env as vec_env_sb
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from torch import nn

import jaxgym.jax.pytree_space as spaces
import jaxsim.typing as jtp
from jaxgym.jax import JaxDataclassEnv, JaxDataclassWrapper, JaxEnv, PyTree
from jaxgym.vector.jax import FlattenSpacesVecWrapper, JaxVectorEnv
from jaxgym.wrappers.jax import (
    ActionNoiseWrapper,
    ClipActionWrapper,
    FlattenSpacesWrapper,
    JaxTransformWrapper,
    NaNHandlerWrapper,
    SquashActionWrapper,
    TimeLimit,
    ToNumPyWrapper,
)
from jaxsim import JaxSim
from jaxsim.physics.algos.soft_contacts import SoftContactsParams
from jaxsim.simulation import simulator_callbacks
from jaxsim.simulation.ode_integration import IntegratorType
from jaxsim.simulation.simulator import SimulatorData, VelRepr
from jaxsim.utils import JaxsimDataclass, Mutability

warnings.simplefilter(action="ignore", category=FutureWarning)


@jax_dataclasses.pytree_dataclass
class ErgoCubObservation(JaxsimDataclass):
    """Observation of the ErgoCub environment."""

    base_height: jtp.Float
    gravity_projection: jtp.Array

    joint_positions: jtp.Array
    joint_velocities: jtp.Array

    base_linear_velocity: jtp.Array
    base_angular_velocity: jtp.Array

    contact_state: jtp.Array

    @staticmethod
    def build(
        base_height: jtp.Float,
        gravity_projection: jtp.Array,
        joint_positions: jtp.Array,
        joint_velocities: jtp.Array,
        base_linear_velocity: jtp.Array,
        base_angular_velocity: jtp.Array,
        contact_state: jtp.Array,
    ) -> "ErgoCubObservation":
        """Build an ErgoCubObservation object."""

        return ErgoCubObservation(
            base_height=jnp.array(base_height, dtype=float),
            gravity_projection=jnp.array(gravity_projection, dtype=float),
            joint_positions=jnp.array(joint_positions, dtype=float),
            joint_velocities=jnp.array(joint_velocities, dtype=float),
            base_linear_velocity=jnp.array(base_linear_velocity, dtype=float),
            base_angular_velocity=jnp.array(base_angular_velocity, dtype=float),
            contact_state=jnp.array(contact_state, dtype=bool),
        )


@dataclasses.dataclass
class MeshcatVizRenderState:
    """Render state of a meshcat-viz visualizer."""

    world: MeshcatWorld = dataclasses.dataclass(init=False)

    _gui_process: Optional[multiprocessing.Process] = dataclasses.field(
        default=None, init=False, repr=False, hash=False, compare=False
    )

    _jaxsim_to_meshcat_viz_name: dict[str, str] = dataclasses.field(
        default_factory=dict, init=False, repr=False, hash=False, compare=False
    )

    def __post_init__(self) -> None:
        """"""

        self.world = MeshcatWorld()
        self.world.open()

    def close(self) -> None:
        """"""

        if self.world is not None:
            self.world.close()

        if self._gui_process is not None:
            self._gui_process.terminate()
            self._gui_process.close()

    @staticmethod
    def open_window(web_url: str) -> None:
        """Open a new window with the given web url."""

        import webview

        print(web_url)
        webview.create_window("meshcat", web_url)
        webview.start(gui="qt")

    def open_window_in_process(self) -> None:
        """"""

        if self._gui_process is not None:
            self._gui_process.terminate()
            self._gui_process.close()

        self._gui_process = multiprocessing.Process(
            target=MeshcatVizRenderState.open_window, args=(self.world.web_url,)
        )
        self._gui_process.start()


StateType = dict[str, SimulatorData | jtp.Array]
ActType = jnp.ndarray
ObsType = ErgoCubObservation
RewardType = float | jnp.ndarray
TerminalType = bool | jnp.ndarray
RenderStateType = MeshcatVizRenderState


@jax_dataclasses.pytree_dataclass
class ErgoCubWalkFuncEnvV0(
    JaxDataclassEnv[
        StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType
    ]
):
    """ErgoCub environment implementing a target reaching task."""

    name: ClassVar = jax_dataclasses.static_field(default="ErgoCubWalkFuncEnvV0")

    # Store an instance of the JaxSim simulator.
    # It gets initialized with SimulatorData with a functional approach.
    _simulator: JaxSim = jax_dataclasses.field(default=None)

    def __post_init__(self) -> None:
        """Environment initialization."""

        model = self.jaxsim.get_model(model_name="ErgoCub")

        # Create the action space (static attribute)
        with self.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            high = jnp.array([25.0] * model.dofs(), dtype=float)

        with self.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            self._action_space = spaces.PyTree(low=-high, high=high)

        # Get joint limits
        s_min, s_max = model.joint_limits()
        s_range = s_max - s_min

        low = ErgoCubObservation.build(
            base_height=0.25,
            gravity_projection=-jnp.ones(3),
            joint_positions=s_min,
            joint_velocities=-50.0 * jnp.ones_like(s_min),
            base_linear_velocity=-5.0 * jnp.ones(3),
            base_angular_velocity=-10.0 * jnp.ones(3),
            contact_state=jnp.array([False] * 4),
        )

        high = ErgoCubObservation.build(
            base_height=1.0,
            gravity_projection=jnp.ones(3),
            joint_positions=s_max,
            joint_velocities=50.0 * jnp.ones_like(s_max),
            base_linear_velocity=5.0 * jnp.ones(3),
            base_angular_velocity=10.0 * jnp.ones(3),
            contact_state=jnp.array([True] * 4),
        )

        # Create the observation space (static attribute)
        with self.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            self._observation_space = spaces.PyTree(low=low, high=high)

    @property
    def jaxsim(self) -> JaxSim:
        """"""

        if self._simulator is not None:
            return self._simulator

        # Create the jaxsim simulator.
        simulator = JaxSim.build(
            # Note: any change of either 'step_size' or 'steps_per_run' requires
            # updating the number of integration steps in the 'transition' method.
            step_size=0.000_250,
            steps_per_run=1,
            velocity_representation=VelRepr.Body,
            integrator_type=IntegratorType.EulerSemiImplicit,
            simulator_data=SimulatorData(
                gravity=jnp.array([0, 0, -10.0]),
                contact_parameters=SoftContactsParams.build(K=10_000, D=20),
            ),
        ).mutable(mutable=True, validate=False)

        # Get the SDF path
        model_sdf_path = resolve_robotics_uri(
            "package://ergoCub/robots/ergoCubGazeboV1_minContacts/model.urdf"
        )

        # Insert the model
        _ = simulator.insert_model_from_description(
            model_description=model_sdf_path, model_name="ErgoCub"
        )

        simulator.data.models = {
            model_name: jax.tree_util.tree_map(lambda leaf: jnp.array(leaf), model_data)
            for model_name, model_data in simulator.data.models.items()
        }

        with self.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            self._simulator = simulator.mutable(mutable=True, validate=True)

        return self._simulator

    def initial(self, rng: Any = None) -> StateType:
        """"""
        assert jax.dtypes.issubdtype(rng, jax.dtypes.prng_key)

        # Split the key
        subkey1, subkey2 = jax.random.split(rng, num=2)

        # Sample an initial observation
        initial_observation: ErgoCubObservation = (
            self.observation_space.sample_with_key(key=subkey1)
        )

        # Sample a goal position
        goal_xy_position = jax.random.uniform(
            key=subkey2, minval=-5.0, maxval=5.0, shape=(2,)
        )

        with self.jaxsim.editable(validate=False) as simulator:
            # Reset the simulator and get the model
            simulator.reset(remove_models=False)
            model = simulator.get_model(model_name="ErgoCub")

            # Reset the joint positions
            model.reset_joint_positions(
                positions=initial_observation.joint_positions,
                joint_names=model.joint_names(),
            )

            # Reset the base position
            model.reset_base_position(position=jnp.array([0, 0, 0.5]))

            # Reset the base velocity
            model.reset_base_velocity(
                base_velocity=jnp.hstack(
                    [
                        0.1 * initial_observation.base_linear_velocity,
                        0.1 * initial_observation.base_angular_velocity,
                    ]
                )
            )

        # Return the simulation state
        return dict(
            simulator_data=simulator.data,
            goal=jnp.array(goal_xy_position, dtype=float),
        )

    def transition(
        self, state: StateType, action: ActType, rng: Any = None
    ) -> StateType:
        """"""

        # Get the JaxSim simulator
        simulator = self.jaxsim

        # Initialize the simulator with the environment state (containing SimulatorData)
        with simulator.editable(validate=True) as simulator:
            simulator.data = state["simulator_data"]

        @jax_dataclasses.pytree_dataclass
        class SetTorquesOverHorizon(simulator_callbacks.PreStepCallback):
            def pre_step(self, sim: JaxSim) -> JaxSim:
                """"""

                model = sim.get_model(model_name="ErgoCub")
                model.zero_input()
                model.set_joint_generalized_force_targets(
                    forces=jnp.atleast_1d(action), joint_names=model.joint_names()
                )

                return sim

        number_of_integration_steps = 40  # 0.010  # TODO 20 for having 0.010

        # Stepping logic
        with simulator.editable(validate=True) as simulator:
            simulator, _ = simulator.step_over_horizon(
                horizon_steps=number_of_integration_steps,
                clear_inputs=False,
                callback_handler=SetTorquesOverHorizon(),
            )

        # Return the new environment state (updated SimulatorData)
        return state | dict(simulator_data=simulator.data)

    def observation(self, state: StateType) -> ObsType:
        """"""

        # Initialize the simulator with the environment state (containing SimulatorData)
        # and get the simulated model
        with self.jaxsim.editable(validate=True) as simulator:
            simulator.data = state["simulator_data"]
            model = simulator.get_model("ErgoCub")

        # Compute the normalized gravity projection in the body frame
        W_R_B = model.base_orientation(dcm=True)
        W_gravity = self.jaxsim.gravity()
        B_gravity = W_R_B.T @ (W_gravity / jnp.linalg.norm(W_gravity))

        W_p_B = model.base_position()
        W_p_goal = jnp.hstack([state["goal"].squeeze(), 0])

        # Compute the distance between the base and the goal in the body frame
        B_p_distance = W_R_B.T @ (W_p_goal - W_p_B)

        # Build the observation from the state
        return ErgoCubObservation.build(
            base_height=model.base_position()[2],
            gravity_projection=B_gravity,
            joint_positions=model.joint_positions(),
            joint_velocities=model.joint_velocities(),
            base_linear_velocity=model.base_velocity()[0:3],
            base_angular_velocity=model.base_velocity()[3:6],
            contact_state=model.in_contact(
                link_names=[name for name in model.link_names() if "_ankle" in name]
            ),
        )

    def reward(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> RewardType:
        """"""

        with self.jaxsim.editable(validate=True) as simulator_next:
            simulator_next.data = next_state["simulator_data"]
            model_next = simulator_next.get_model("ErgoCub")

        terminal = self.terminal(state=state)
        obs_in_space = jax.lax.select(
            pred=self.observation_space.contains(x=self.observation(state=state)),
            on_true=1.0,
            on_false=0.0,
        )

        # Position of the base
        W_p_B = model_next.base_position()
        W_p_xy_goal = state["goal"]

        reward = 0.0
        reward += 1.0 * (1.0 - jnp.array(terminal, dtype=float))  # alive
        reward += 5.0 * obs_in_space  #
        # reward += 100.0 * v_WB[0]  # forward velocity
        reward -= jnp.linalg.norm(W_p_B[0:2] - W_p_xy_goal)  # distance from goal
        reward += 1.0 * model_next.in_contact(
            link_names=[
                name
                for name in model_next.link_names()
                if name.startswith("leg_") and name.endswith("_lower")
            ]
        ).any().astype(float)
        reward -= 0.1 * jnp.linalg.norm(action) / action.size  # control cost

        return reward

    def terminal(self, state: StateType) -> TerminalType:
        # Get the current observation
        observation = self.observation(state=state)

        base_too_high = (
            observation.base_height >= self.observation_space.high.base_height
        )
        return base_too_high

    # =========
    # Rendering
    # =========

    def render_image(
        self, state: StateType, render_state: RenderStateType
    ) -> tuple[RenderStateType, npt.NDArray]:
        """Show the state."""

        model_name = "ErgoCub"

        # Initialize the simulator with the environment state (containing SimulatorData)
        # and get the simulated model
        with self.jaxsim.editable(validate=False) as simulator:
            simulator.data = state["simulator_data"]
            model = simulator.get_model(model_name=model_name)

        # Insert the model lazily in the visualizer if it is not already there
        if model_name not in render_state.world._meshcat_models.keys():
            from rod.urdf.exporter import UrdfExporter

            urdf_string = UrdfExporter.sdf_to_urdf_string(
                sdf=rod.Sdf(
                    version="1.7",
                    model=model.physics_model.description.extra_info["sdf_model"],
                ),
                pretty=True,
                gazebo_preserve_fixed_joints=False,
            )

            meshcat_viz_name = render_state.world.insert_model(
                model_description=urdf_string, is_urdf=True, model_name=None
            )

            render_state._jaxsim_to_meshcat_viz_name[model_name] = meshcat_viz_name

        # Check that the model is in the visualizer
        if (
            not render_state._jaxsim_to_meshcat_viz_name[model_name]
            in render_state.world._meshcat_models.keys()
        ):
            raise ValueError(f"The '{model_name}' model is not in the meshcat world")

        # Update the model in the visualizer
        render_state.world.update_model(
            model_name=render_state._jaxsim_to_meshcat_viz_name[model_name],
            joint_names=model.joint_names(),
            joint_positions=model.joint_positions(),
            base_position=model.base_position(),
            base_quaternion=model.base_orientation(dcm=False),
        )

        return render_state, np.empty(0)

    def render_init(self, open_gui: bool = False, **kwargs) -> RenderStateType:
        """Initialize the render state."""

        # Initialize the render state
        meshcat_viz_state = MeshcatVizRenderState()

        if open_gui:
            meshcat_viz_state.open_window_in_process()

        return meshcat_viz_state

    def render_close(self, render_state: RenderStateType) -> None:
        """Close the render state."""

        render_state.close()


class ErgoCubWalkEnvV0(JaxEnv):
    """"""

    def __init__(self, render_mode: str | None = None, **kwargs: Any) -> None:
        """"""

        from jaxgym.wrappers.jax import (
            ClipActionWrapper,
            FlattenSpacesWrapper,
            JaxTransformWrapper,
            TimeLimit,
        )

        func_env = ErgoCubWalkFuncEnvV0()

        func_env_wrapped = func_env
        func_env_wrapped = TimeLimit(
            env=func_env_wrapped, max_episode_steps=5_000
        )  # TODO
        func_env_wrapped = ClipActionWrapper(env=func_env_wrapped)
        func_env_wrapped = FlattenSpacesWrapper(env=func_env_wrapped)
        func_env_wrapped = JaxTransformWrapper(env=func_env_wrapped, function=jax.jit)

        super().__init__(
            func_env=func_env_wrapped,
            metadata=self.metadata,
            render_mode=render_mode,
        )


class ErgoCubWalkVectorEnvV0(JaxVectorEnv):
    """"""

    metadata = dict()

    def __init__(
        self,
        # func_env: JaxDataclassEnv[
        #     StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType
        # ],
        num_envs: int,
        render_mode: str | None = None,
        # max_episode_steps: int = 5_000,
        jit_compile: bool = True,
        **kwargs,
    ) -> None:
        """"""

        print("+++", kwargs)

        env = ErgoCubWalkFuncEnvV0()

        # Vectorize the environment.
        # Note: it automatically wraps the environment in a TimeLimit wrapper.
        super().__init__(
            func_env=env,
            num_envs=num_envs,
            metadata=self.metadata,
            render_mode=render_mode,
            max_episode_steps=5_000,  # TODO
            jit_compile=jit_compile,
        )

        # from jaxgym.vector.jax import FlattenSpacesVecWrapper
        #
        # vec_env_wrapped = FlattenSpacesVecWrapper(env=vec_env)


if __name__ == "__main__":
    """Stable Baselines"""

    def make_jax_env(
        max_episode_steps: Optional[int] = 500, jit: bool = True
    ) -> JaxEnv:
        """"""

        # TODO: single env -> time limit with stable_baselines?

        if max_episode_steps in {None, 0}:
            env = ErgoCubWalkFuncEnvV0()
        else:
            env = TimeLimit(
                env=ErgoCubWalkFuncEnvV0(), max_episode_steps=max_episode_steps
            )

        return JaxEnv(
            func_env=ToNumPyWrapper(
                env=FlattenSpacesWrapper(env=env)
                if not jit
                else JaxTransformWrapper(
                    function=jax.jit,
                    env=FlattenSpacesWrapper(env=env),
                ),
            ),
            render_mode="meshcat_viz",
        )

    class CustomVecEnvSB(vec_env_sb.VecEnv):
        """"""

        metadata = {"render_modes": []}

        def __init__(
            self,
            jax_vector_env: JaxVectorEnv | VectorWrapper,
            log_rewards: bool = False,
            # num_envs: int,
            # observation_space: spaces.Space,
            # action_space: spaces.Space,
            # render_mode: Optional[str] = None,
        ) -> None:
            """"""

            if not isinstance(jax_vector_env.unwrapped, JaxVectorEnv):
                raise TypeError(type(jax_vector_env))

            self.jax_vector_env = jax_vector_env

            single_env_action_space: PyTree = (
                jax_vector_env.unwrapped.single_action_space
            )

            single_env_observation_space: PyTree = (
                jax_vector_env.unwrapped.single_observation_space
            )

            super().__init__(
                num_envs=self.jax_vector_env.num_envs,
                action_space=single_env_action_space.to_box(),
                observation_space=single_env_observation_space.to_box(),
            )

            self.actions = np.zeros_like(self.jax_vector_env.action_space.sample())

            # Initialize the RNG seed
            self._seed = None
            self.seed()

            # Initialize the rewards logger
            self.logger_rewards = [] if log_rewards else None

        def reset(self) -> vec_env_sb.base_vec_env.VecEnvObs:
            """"""

            observations, state_infos = self.jax_vector_env.reset(seed=self._seed)
            return np.array(observations)

        def step_async(self, actions: np.ndarray) -> None:
            self.actions = actions

        @staticmethod
        @functools.partial(jax.jit, static_argnames=("batch_size",))
        def tree_inverse_transpose(
            pytree: jtp.PyTree, batch_size: int
        ) -> List[jtp.PyTree]:
            """"""

            return [
                jax.tree_util.tree_map(lambda leaf: leaf[i], pytree)
                for i in range(batch_size)
            ]

        def step_wait(self) -> vec_env_sb.base_vec_env.VecEnvStepReturn:
            """"""

            (
                observations,
                rewards,
                terminals,
                truncated,
                step_infos,
            ) = self.jax_vector_env.step(actions=self.actions)

            done = np.logical_or(terminals, truncated)

            # list_of_step_infos = [
            #     jax.tree_util.tree_map(lambda l: l[i], step_infos)
            #     for i in range(self.jax_vector_env.num_envs)
            # ]

            list_of_step_infos = self.tree_inverse_transpose(
                pytree=step_infos, batch_size=self.jax_vector_env.num_envs
            )

            # def pytree_to_numpy(pytree: jtp.PyTree) -> jtp.PyTree:
            #     return jax.tree_util.tree_map(lambda leaf: np.array(leaf), pytree)
            #
            # list_of_step_infos_numpy = [pytree_to_numpy(pt) for pt in list_of_step_infos]

            list_of_step_infos_numpy = [
                ToNumPyWrapper.pytree_to_numpy(pytree=pt) for pt in list_of_step_infos
            ]

            if self.logger_rewards is not None:
                self.logger_rewards.append(np.array(rewards).mean())

            return (
                np.array(observations),
                np.array(rewards),
                np.array(done),
                list_of_step_infos_numpy,
            )

        def close(self) -> None:
            return self.jax_vector_env.close()

        def get_attr(
            self, attr_name: str, indices: vec_env_sb.base_vec_env.VecEnvIndices = None
        ) -> List[Any]:
            raise AttributeError
            # raise NotImplementedError

        def set_attr(
            self,
            attr_name: str,
            value: Any,
            indices: vec_env_sb.base_vec_env.VecEnvIndices = None,
        ) -> None:
            raise NotImplementedError

        def env_method(
            self,
            method_name: str,
            *method_args,
            indices: vec_env_sb.base_vec_env.VecEnvIndices = None,
            **method_kwargs,
        ) -> List[Any]:
            raise NotImplementedError

        def env_is_wrapped(
            self,
            wrapper_class: Type[gym.Wrapper],
            indices: vec_env_sb.base_vec_env.VecEnvIndices = None,
        ) -> List[bool]:
            return [False] * self.num_envs
            # raise NotImplementedError

        def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
            """"""

            if seed is None:
                seed = np.random.default_rng().integers(0, 2 ** 32 - 1, dtype="uint32")

            if np.array(seed, dtype="uint32") != np.array(seed):
                raise ValueError(f"seed must be compatible with 'uint32' casting")

            self._seed = seed
            return [seed]

            # _ = self.jax_vector_env.reset(seed=seed)
            # return [None]

    def make_vec_env_stable_baselines(
        jax_dataclass_env: JaxDataclassEnv | JaxDataclassWrapper,
        n_envs: int = 1,
        seed: Optional[int] = None,
        vec_env_kwargs: Optional[Dict[str, Any]] = None,
    ) -> vec_env_sb.VecEnv:
        """"""

        env = jax_dataclass_env

        vec_env_kwargs = vec_env_kwargs if vec_env_kwargs is not None else dict()

        vec_env = JaxVectorEnv(
            func_env=env,
            num_envs=n_envs,
            **vec_env_kwargs,
        )

        # Flatten the PyTree spaces to regular Box spaces
        vec_env = FlattenSpacesVecWrapper(env=vec_env)

        vec_env_sb = CustomVecEnvSB(jax_vector_env=vec_env, log_rewards=True)

        if seed is not None:
            _ = vec_env_sb.seed(seed=seed)

        return vec_env_sb

    os.environ["IGN_GAZEBO_RESOURCE_PATH"] = "/conda/share/"  # DEBUG

    max_episode_steps = 200
    func_env = NaNHandlerWrapper(env=ErgoCubWalkFuncEnvV0())

    if max_episode_steps is not None:
        func_env = TimeLimit(env=func_env, max_episode_steps=max_episode_steps)

    func_env = ClipActionWrapper(
        env=SquashActionWrapper(env=ActionNoiseWrapper(env=func_env)),
    )

    vec_env = make_vec_env_stable_baselines(
        jax_dataclass_env=func_env,
        n_envs=6000,
        seed=42,
        vec_env_kwargs=dict(
            jit_compile=True,
        ),
    )

    vec_env = VecMonitor(
        venv=VecNormalize(
            venv=vec_env,
            training=True,
        )
    )

    vec_env.venv.venv.logger_rewards = []
    seed = vec_env.seed(seed=7)[0]
    _ = vec_env.reset()

    model = PPO(
        "MlpPolicy",
        env=vec_env,
        n_steps=5,  # in the vector env -> real ones are x512
        batch_size=256,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=0.1,
        normalize_advantage=True,
        target_kl=0.025,
        verbose=2,
        learning_rate=0.000_300,
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=dict(pi=[512, 512], vf=[512, 512]),
            log_std_init=np.log(0.05),
        ),
    )

    print(model.policy)

    model = model.learn(total_timesteps=50000, progress_bar=True)
