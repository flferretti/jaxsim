import mujoco as mj
import numpy as np

from .loaders import RodModelToMjcf, SdfToMjcf, UrdfToMjcf
from .model import MujocoModelHelper
from .visualizer import MujocoVideoRecorder, MujocoVisualizer


def mujoco_data_from_jaxsim(
    mujoco_model: mj.MjModel,
    jaxsim_model,
    jaxsim_data,
    mujoco_data: mj.MjData | None = None,
) -> mj.MjData:
    """"""

    import jaxsim.api as js

    if not isinstance(jaxsim_model, js.model.JaxSimModel):
        raise ValueError("The `jaxsim_model` argument must be a JaxSimModel object.")

    if not isinstance(jaxsim_data, js.data.JaxSimModelData):
        raise ValueError("The `jaxsim_data` argument must be a JaxSimModelData object.")

    model_helper = MujocoModelHelper(model=mujoco_model, data=mujoco_data)

    # Set the model position.
    model_helper.set_base_position(position=np.array(jaxsim_data.base_position()))

    # Set the model orientation.
    model_helper.set_base_orientation(
        orientation=np.array(jaxsim_data.base_orientation())
    )

    # Set the joint positions.
    if jaxsim_model.dofs() > 0:

        model_helper.set_joint_positions(
            joint_names=list(jaxsim_model.joint_names()),
            positions=np.array(
                jaxsim_data.joint_positions(
                    model=jaxsim_model, joint_names=jaxsim_model.joint_names()
                )
            ),
        )

    # Create a dictionary with the joints that have been removed for various reasons
    # (mainly after model reduction).
    joints_removed_dict = {
        j.name: j
        for j in jaxsim_model.description._joints_removed
        if j.name not in set(jaxsim_model.joint_names())
    }

    # Get all the joints in the Mujoco and JaxSim models.
    joints_mujoco = set(model_helper.joint_names())

    # Set the positions of the removed joints.
    for joint_name in set(joints_removed_dict.keys()):

        if joint_name not in joints_mujoco:
            continue

        _ = [
            model_helper.set_joint_position(
                position=joints_removed_dict[joint_name].initial_position,
                joint_name=joint_name,
            )
        ]

    mj.mj_forward(mujoco_model, model_helper.data)
    return model_helper.data
