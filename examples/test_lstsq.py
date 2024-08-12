import pathlib
import time
from typing import Any

import jax
import jax.numpy as jnp
import jaxopt
import loop_rate_limiters
import numpy as np
import rod
import rod.builder.primitives
import rod.urdf.exporter

import jaxsim
import jaxsim.api as js
import jaxsim.exceptions
import jaxsim.mujoco
import jaxsim.typing as jtp
from jaxsim import VelRepr

# ==========
# Simple box
# ==========

# Construct the model programmatically.
rod_sdf_model = (
    rod.builder.primitives.BoxBuilder(x=0.3, y=0.2, z=0.1, mass=10.0, name="box")
    .build_model()
    .add_link()
    .add_inertial()
    .add_visual()
    .add_collision()
    .build()
)
urdf_path = rod.urdf.exporter.UrdfExporter(pretty=True).to_urdf_string(
    sdf=rod_sdf_model
)

# Build the model.
model = js.model.JaxSimModel.build_from_model_description(
    model_description=rod_sdf_model,
)

# ====================================
# Create and initialize the integrator
# ====================================

# Initialize the data of the model.
data_t0 = js.data.JaxSimModelData.build(
    model=model,
    base_position=jnp.array([0, 0, 2.0]),
    # base_quaternion=jnp.array([0, 1.0, 0, 0]),
    # contacts_params=js.contact.estimate_good_soft_contacts_parameters(
    #     model=model, number_of_active_collidable_points_steady_state=4
    # ),
)

# # Create the integrator.
# integrator = jaxsim.integrators.fixed_step.RungeKutta4SO3.build(
#     fsal_enabled_if_supported=False,
#     dynamics=js.ode.wrap_system_dynamics_for_integration(
#         model=model,
#         data=data_t0,
#         system_dynamics=functools.partial(
#             js.ode.system_dynamics, baumgarte_quaternion_regularization=1.0
#         ),
#     ),
# )

# Initialize the integration horizon.
t0, tf, dt = 0.0, 1.0, 0.001
t_ns = jnp.arange(start=t0, stop=tf * 1e9, step=dt * 1e9).astype(int)
t = jnp.array(t_ns / 1e9).astype(float)

# # Initialize the integrator state.
# integrator_state_t0 = integrator.init(x0=data_t0.state, t0=t0, dt=dt)

# =============================
# Initialize the contact status
# =============================

# Hardcode the active collidable point.
# This should be replaced by contact detection.
active_collidable_points = (
    jnp.zeros_like(jnp.array(model.kin_dyn_parameters.contact_parameters.body)).astype(
        bool
    )
    # .at[0:4]
    .at[0]
    # .at[1]
    # .at[0:2]
    # .at[0:3]
    .set(True)
)

# Get the number of total and active collidable points.
num_of_collidable_points = int(active_collidable_points.size)
num_of_active_collidable_points = int(active_collidable_points.astype(int).sum())


# ============================================
# Functions to compute explicit contact forces
# ============================================


@jax.jit
def delassus_and_B_matrix(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jax.Array:
    """"""

    M = js.model.free_floating_mass_matrix(model=model, data=data)
    h = js.model.free_floating_bias_forces(model=model, data=data)

    sl = jnp.s_[active_collidable_points, 0:3, :]
    J_WC = jnp.vstack(js.contact.jacobian(model=model, data=data)[sl])

    J̇_WC = jnp.vstack(js.contact.jacobian_derivative(model=model, data=data)[sl])
    ν = data.generalized_velocity()

    G = J_WC @ jnp.linalg.lstsq(M, J_WC.T)[0]
    b = J_WC @ jnp.linalg.lstsq(M, -h)[0] + J̇_WC @ ν

    return G, b


@jax.jit
def compute_constraints_forces(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    CW_al_free_WC: jtp.MatrixLike,
) -> jtp.Matrix:
    """"""

    # Compute the Delassus matrix.
    with data.switch_velocity_representation(VelRepr.Mixed):
        G, b = delassus_and_B_matrix(model=model, data=data)  # noqa: F841

    # diff = b - CW_al_free_WC.squeeze()
    # *PASSED* assert jnp.allclose(diff, 0.0), diff

    # Compute the constrained acceleration of the collidable points.
    # Note that we operate on mixed accelerations of collidable points.
    objective = lambda x: jnp.sum(
        jnp.square(G @ x - jnp.vstack(CW_al_free_WC.flatten()))
    )
    # objective = lambda x: jnp.linalg.norm(G @ x - jnp.vstack(CW_al_free_WC.flatten()))

    # Compute the 3D linear force in C[W] frame

    # opt = jaxopt.ProjectedGradient(
    #     fun=objective,
    #     projection=jaxopt.projection.projection_non_negative,
    #     maxiter=150,
    #     implicit_diff=False,
    #     maxls=20,
    #     tol=1e-8,
    #     # verbose=2,
    # )

    opt = jaxopt.GradientDescent(
        fun=objective,
        maxiter=20000,
        tol=1e-10,
        stepsize=0.001,
        maxls=50,
        # verbose=2,
    )

    # opt = jaxopt.LBFGS(
    #     fun=objective,
    #     maxiter=1000,
    #     tol=1e-10,
    #     maxls=100,
    #     history_size=10,
    #     max_stepsize=100.0,
    #     stop_if_linesearch_fails=True,
    #     # verbose=2,
    # )

    CW_fl_constraints_flat_OPT = (
        opt.run(
            init_params=jnp.zeros_like(jnp.vstack(CW_al_free_WC.flatten()))
        )  # , hyperparams_prox=None
        .params.reshape(-1, 3)
        .squeeze()
    )
    # CW_f_Ci6D_OPT = jnp.zeros(shape=(sl.shape[0], 6)).at[:, :3].set(CW_fl_constraints_flat_OPT)

    # Unpack the linear forces that enforce the constraints.
    CW_fl_constraints_OPT = jnp.vstack(
        jnp.split(CW_fl_constraints_flat_OPT, num_of_active_collidable_points)
    )

    CW_fl_constraints_flat_LSTSQ = jnp.linalg.lstsq(
        G, -jnp.vstack(CW_al_free_WC.flatten())
    )[0].squeeze()

    # # Unpack the linear forces that enforce the constraints.
    # CW_fl_constraints_LSTSQ = jnp.vstack(
    #     jnp.split(CW_fl_constraints_flat_LSTSQ, num_of_active_collidable_points)
    # )

    diff = CW_fl_constraints_flat_LSTSQ - CW_fl_constraints_flat_OPT
    # !FAILS! assert jnp.allclose(diff, 0.0), diff

    # Note that we return the mixed linear forces of collidable points.
    return CW_fl_constraints_OPT, diff


@jax.jit
def linear_acceleration_of_collidable_points(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    collidable_point_indices: jax.Array,
    references: js.references.JaxSimModelReferences | None = None,
) -> jax.Array:
    """"""

    references = (
        references
        if references is not None
        else js.references.JaxSimModelReferences.zero(model=model)
    )

    with (
        data.switch_velocity_representation(VelRepr.Body),
        references.switch_velocity_representation(data.velocity_representation),
    ):

        W_v̇_WB, s̈, *_ = js.ode.system_velocity_dynamics(
            model=model,
            data=data,
            joint_forces=references.joint_force_references(model=model),
            link_forces=references.link_forces(model=model, data=data),
        )
        W_ν̇ = jnp.hstack([W_v̇_WB, s̈])

    with data.switch_velocity_representation(VelRepr.Inertial):
        O_J_WL_W = js.contact.jacobian(
            model=model,
            data=data,
            output_vel_repr=VelRepr.Mixed,
        )

    with data.switch_velocity_representation(VelRepr.Inertial):
        W_ν = data.generalized_velocity()
        O_J̇_WL_W = js.contact.jacobian_derivative(
            model=model,
            data=data,
            output_vel_repr=VelRepr.Mixed,
        )

    O_v̇_WC = jax.vmap(
        lambda idx: O_J̇_WL_W[idx, :, :] @ W_ν + O_J_WL_W[idx, :, :] @ W_ν̇
    )(collidable_point_indices)

    return O_v̇_WC[:, 0:3]


@jax.jit
def compute_link_forces_inertial_fixed(
    CW_fl_constraints: jax.Array, data: js.data.JaxSimModelData
) -> jax.Array:
    """"""

    W_H_C = js.contact.transforms(model=model, data=data)[
        active_collidable_points, :, :
    ]

    def mixed_to_inertial(W_H_C: jax.Array, CW_fl: jax.Array) -> jax.Array:
        W_Xf_CW = jaxsim.math.Adjoint.from_transform(
            W_H_C.at[0:3, 0:3].set(jnp.eye(3)),
            inverse=True,
        ).T
        return W_Xf_CW @ jnp.hstack([CW_fl, jnp.zeros(3)])

    W_f_active_C = jax.vmap(mixed_to_inertial)(W_H_C, CW_fl_constraints)

    W_f_C = (
        jnp.zeros(shape=(num_of_collidable_points, 6))
        .at[active_collidable_points, :]
        .set(W_f_active_C)
    )

    parent_link_index_of_collidable_points = jnp.array(
        model.kin_dyn_parameters.contact_parameters.body
    )

    mask = parent_link_index_of_collidable_points[:, jnp.newaxis] == jnp.arange(
        model.number_of_links()
    )

    W_f_L = mask.T @ W_f_C

    return W_f_L


# ===============================================================================
# Functions to compute with AD the linear mixed acceleration of collidable points
# ===============================================================================


def mixed_linear_velocity_of_collidable_point(
    q: jax.Array,
    ν: jax.Array,  # This must be mixed
    collidable_point_index: jax.Array | int,
) -> jax.Array:
    """"""

    data_ad = js.data.JaxSimModelData.zero(
        model=model, velocity_representation=VelRepr.Mixed
    )

    data_ad = data_ad.reset_base_position(base_position=q[0:3])
    data_ad = data_ad.reset_base_quaternion(base_quaternion=q[3:7])
    data_ad = data_ad.reset_joint_positions(positions=q[7:], model=model)

    # TODO: note the hardcoded assumption of Mixed
    data_ad.reset_joint_velocities(velocities=ν[6:], model=model)
    data_ad = data_ad.reset_base_velocity(
        base_velocity=ν[0:6], velocity_representation=VelRepr.Mixed
    )

    W_ṗ_C = js.contact.collidable_point_velocities(model=model, data=data_ad)[
        collidable_point_index, :
    ]

    return W_ṗ_C


def compute_q(data: js.data.JaxSimModelData) -> jax.Array:
    """"""

    q = jnp.hstack(
        (
            data.base_position(),
            data.base_orientation(),
            data.joint_positions(model=model),
        )
    )

    return q


def compute_q̇(data: js.data.JaxSimModelData) -> jax.Array:
    """"""

    with data.switch_velocity_representation(VelRepr.Body):
        B_ω_WB = data.base_velocity()[3:6]

    with data.switch_velocity_representation(VelRepr.Mixed):
        W_ṗ_B = data.base_velocity()[0:3]

    W_Q̇_B = jaxsim.math.Quaternion.derivative(
        quaternion=data.base_orientation(),
        omega=B_ω_WB,
        omega_in_body_fixed=True,
        K=0.0,
    ).squeeze()

    q̇ = jnp.hstack([W_ṗ_B, W_Q̇_B, data.joint_velocities()])

    return q̇


def compute_ν(data: js.data.JaxSimModelData) -> jax.Array:
    """"""

    ν = jnp.hstack(
        (
            data.base_velocity(),
            data.joint_velocities(),
        )
    )

    return ν


@jax.jit
def linear_acceleration_of_collidable_points_ad(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    collidable_point_indices: jax.Array,
) -> jax.Array:

    q = compute_q(data=data)
    q̇ = compute_q̇(data=data)

    with data.switch_velocity_representation(VelRepr.Mixed):
        ν = compute_ν(data=data)
        ν̇ = jnp.hstack(js.model.forward_dynamics_aba(model=model, data=data))

    def linear_velocity_of_collidable_point(
        collidable_point_index: jax.Array | int,
    ) -> jax.Array:

        df_dq = jax.jacrev(mixed_linear_velocity_of_collidable_point, argnums=0)(
            q, ν, collidable_point_index
        )

        df_dν = jax.jacrev(mixed_linear_velocity_of_collidable_point, argnums=1)(
            q, ν, collidable_point_index
        )

        CW_a_ad_WC = df_dq @ q̇ + df_dν @ ν̇

        return CW_a_ad_WC

    return jax.vmap(linear_velocity_of_collidable_point)(collidable_point_indices)


# ===============================
# Custom semi-implicit integrator
# ===============================


@jax.jit
def semi_implicit_forward_euler(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    dt: jtp.FloatLike,
    joint_forces: jtp.VectorLike | None = None,
    link_forces: jtp.MatrixLike | None = None,
) -> tuple[js.data.JaxSimModelData, dict[str, Any]]:
    """"""

    # =====================================
    # Integrate the system velocity (mixed)
    # =====================================

    # # Compute the system acceleration (always in inertial-fixed).
    # W_v̇_WB, s̈, *_ = js.ode.system_velocity_dynamics(
    #     model=model,
    #     data=data,
    #     joint_forces=joint_forces,
    #     link_forces=link_forces,
    # )
    #
    # W_X_BW = jaxsim.math.Adjoint.from_transform(
    #     jnp.eye(4).at[0:3, 3].set(data.base_position())
    # )
    #
    # with data.switch_velocity_representation(VelRepr.Mixed):
    #     BW_v_WB = data.base_velocity()
    #     BW_v_W_BW = jnp.zeros(6).at[0:3].set(BW_v_WB[0:3])
    #     W_v_W_BW = W_X_BW @ BW_v_W_BW
    #
    # with data.switch_velocity_representation(VelRepr.Inertial):
    #     W_v_WB = data.base_velocity()
    #
    # BW_X_W = jaxsim.math.Adjoint.inverse(W_X_BW)
    # BW_v̇_WB = BW_X_W @ (W_v̇_WB - jaxsim.math.Cross.vx(W_v_W_BW) @ W_v_WB)
    #
    # # Integrate the system velocity.
    # BW_v_WB = BW_v_WB + dt * BW_v̇_WB
    # ṡ = data.joint_velocities() + dt * s̈
    #
    # # Create a semi-implicit data object.
    # data_si = data.copy()
    # data_si = data_si.reset_joint_velocities(velocities=ṡ)
    # data_si = data_si.reset_base_velocity(
    #     base_velocity=BW_v_WB, velocity_representation=VelRepr.Mixed
    # )

    # ========================================
    # Integrate the system velocity (inertial)
    # ========================================

    # Compute the system acceleration (always in inertial-fixed).
    W_v̇_WB, s̈, *_ = js.ode.system_velocity_dynamics(
        model=model,
        data=data,
        joint_forces=joint_forces,
        link_forces=link_forces,
    )

    with data.switch_velocity_representation(VelRepr.Inertial):
        W_v_WB = data.base_velocity()

    # Integrate the system velocity.
    W_v_WB = W_v_WB + dt * W_v̇_WB
    ṡ = data.joint_velocities() + dt * s̈

    # Create a semi-implicit data object.
    data_si = data.copy()
    data_si = data_si.reset_joint_velocities(velocities=ṡ)
    data_si = data_si.reset_base_velocity(
        base_velocity=W_v_WB, velocity_representation=VelRepr.Inertial
    )

    # =============================
    # Integrate the system position
    # =============================

    W_ṗ_B_si, _, ṡ = js.ode.system_position_dynamics(
        model=model,
        data=data_si,
        baumgarte_quaternion_regularization=0.0,
    )

    with data_si.switch_velocity_representation(VelRepr.Mixed):
        BW_v_WB = data_si.base_velocity()

    W_p_B = data_si.base_position() + dt * W_ṗ_B_si

    W_Q_B = jaxsim.math.Quaternion.integration(
        quaternion=data_si.base_orientation(),
        dt=dt,
        omega=BW_v_WB[3:6],
        omega_in_body_fixed=False,
    )

    s = data_si.joint_positions() + dt * ṡ

    data_tf = data_si.copy()
    data_tf = data_tf.reset_joint_positions(positions=s)
    data_tf = data_tf.reset_base_position(base_position=W_p_B)
    data_tf = data_tf.reset_base_quaternion(base_quaternion=W_Q_B)

    return data_tf, {}


# ===============
# Simulation loop
# ===============

data = data_t0.copy()
# integrator_state = integrator_state_t0
W_p0_C = js.contact.collidable_point_positions(model=model, data=data)

references = js.references.JaxSimModelReferences.build(
    model=model, velocity_representation=data.velocity_representation
)

s = []
W_p_B = []
W_Q_B = []
diffs = []

for ns in t_ns:

    print(f"\n=======\nt={ns / 1e9:.3f}\n=======\n\n", end="\r", flush=True)

    with data.switch_velocity_representation(VelRepr.Mixed):

        # Compute the free acceleration of the collidable points.
        CW_al_free_WC = linear_acceleration_of_collidable_points(
            model,
            data,
            jnp.where(active_collidable_points)[0],
            None,
        )

        CW_fl_constraints, diff = compute_constraints_forces(
            model=model, data=data, CW_al_free_WC=CW_al_free_WC
        )

    # =================================================================
    # Check that the constraints are satisfied with the computed forces
    # =================================================================

    with references.switch_velocity_representation(VelRepr.Inertial):

        W_f_L = compute_link_forces_inertial_fixed(
            CW_fl_constraints=CW_fl_constraints, data=data
        )

        references = references.apply_link_forces(
            forces=W_f_L, model=model, data=data, additive=False
        )

    # check_constraints = True
    check_constraints = False

    if check_constraints:

        # Test
        with (
            data.switch_velocity_representation(VelRepr.Mixed),
            references.switch_velocity_representation(data.velocity_representation),
        ):

            CW_al_constrained_WC = linear_acceleration_of_collidable_points(
                model,
                data,
                jnp.where(active_collidable_points)[0],
                references,
            )

            # assert jnp.allclose(
            #     CW_al_constrained_WC, jnp.zeros_like(CW_al_constrained_WC), atol=1e-3
            # )

            # print("    CW_al_constr_WC=", CW_al_constrained_WC)
            # print("    CW_al_constr_WC_nano=", CW_al_constrained_WC * 1e9)

    # ===================
    # Step the simulation
    # ===================

    with (
        data.switch_velocity_representation(VelRepr.Inertial),
        references.switch_velocity_representation(data.velocity_representation),
    ):

        # Integrate with the custom integrator.
        data, _ = semi_implicit_forward_euler(
            model=model,
            data=data,
            dt=dt,
            link_forces=references.link_forces(model=model, data=data),
            joint_forces=references.joint_force_references(model=model),
        )

        # # Integrate with one of the JaxSim integrators.
        # data, integrator_state = js.model.step(
        #     model=model,
        #     data=data,
        #     dt=dt,
        #     integrator=integrator,
        #     integrator_state=integrator_state,
        #     link_forces=references.link_forces(model=model, data=data),
        #     joint_forces=references.joint_force_references(model=model),
        # )

        data: js.data.JaxSimModelData
        W_p_B.append(data.base_position())
        W_Q_B.append(data.base_orientation())
        s.append(data.joint_positions(model, joint_names=tuple(model.joint_names())))
        diffs.append(diff)

import matplotlib.pyplot as plt

plt.plot(t, jnp.vstack(diffs), label=["x", "y", "z"])
plt.xlabel("Time [s]")
plt.ylabel("Force error")
plt.title("[GradientDescent] Force Difference")
plt.grid()
plt.legend()
plt.show()

# ================
# 3D visualization
# ================

# raise

mjcf_string, assets = jaxsim.mujoco.RodModelToMjcf.convert(
    rod_model=model.built_from,
    cameras=jaxsim.mujoco.loaders.MujocoCamera.build_from_target_view(
        camera_name="recorder",
        lookat=(0, 0, 1.0),
        distance=5.0,
        azimut=180 - 45,
        elevation=-20,
        fovy=45,
    ),
)

mj_model_helper = jaxsim.mujoco.MujocoModelHelper.build_from_xml(
    mjcf_description=mjcf_string, assets=assets
)

mj_model_helper.data = jaxsim.mujoco.mujoco_data_from_jaxsim(
    mujoco_model=mj_model_helper.model, jaxsim_model=model, jaxsim_data=data_t0
)

viz = jaxsim.mujoco.MujocoVisualizer(
    model=mj_model_helper.model, data=mj_model_helper.data
)

# ================
# 3D visualization
# ================

with viz.open(
    lookat=(0, 0, 1.0), distance=5.0, azimut=180 - 45, elevation=-20
) as viewer:

    rtf = 0.5
    # time.sleep(2.0)
    rate = loop_rate_limiters.RateLimiter(frequency=float(1 / 0.020 * rtf))

    for idx, (ns, _) in enumerate(zip(t_ns, W_p_B, strict=True)):
        if int(ns) % int(0.020 * 1e9) != 0:
            continue

        with viewer.lock():
            mj_model_helper.set_base_position(position=np.array(W_p_B[idx]))
            mj_model_helper.set_base_orientation(orientation=np.array(W_Q_B[idx]))

            if model.dofs() > 0:
                mj_model_helper.set_joint_positions(
                    joint_names=list(model.joint_names()), positions=np.array(s[idx])
                )

        viz.sync(viewer=viewer)
        rate.sleep()

    while viewer.is_running():
        time.sleep(0.500)

# ========
# 3D video
# ========

recorder = jaxsim.mujoco.MujocoVideoRecorder(
    model=mj_model_helper.model,
    data=mj_model_helper.data,
    fps=int(1 / 0.020),  # 50 Hz
    width=320 * 4,
    height=240 * 4,
)

for idx, (ns, _) in enumerate(zip(t_ns, W_p_B, strict=True)):

    if ns % int(0.020 * 1e9) != 0:
        continue

    mj_model_helper.set_base_position(position=np.array(W_p_B[idx]))
    mj_model_helper.set_base_orientation(orientation=np.array(W_Q_B[idx]))
    if model.dofs() > 0:
        mj_model_helper.set_joint_positions(
            joint_names=list(model.joint_names()), positions=np.array(s[idx])
        )
    recorder.record_frame(camera_name="recorder")

recorder.write_video(path=pathlib.Path.home() / "git" / "out.mp4", exist_ok=True)
