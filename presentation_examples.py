import os
import time
from pathlib import Path

import jax.numpy as jnp
import rod
from rod.builder.primitives import BoxBuilder

import jaxsim
import jaxsim.api as js
import jaxsim.mujoco
from jaxsim import VelRepr, logging

os.environ["MUJOCO_GL"] = "egl"


def main():
    # Simulation Parameters.
    # T = 5000
    TIME_STEP = 0.02
    T = int(5 / TIME_STEP)

    # Terrain Parameters.
    TERRAIN = jaxsim.terrain.PlaneTerrain.build(normal=(0.2, -0.1, 0.5))

    CONTACT_MODEL_DICT = {
        "soft": jaxsim.rbda.contacts.SoftContacts,
        "rigid": jaxsim.rbda.contacts.RigidContacts,
        "relaxed": jaxsim.rbda.contacts.RelaxedRigidContacts,
        "viscous": jaxsim.rbda.contacts.ViscoElasticContacts,
    }

    CONTACT_PARAMS_DICT = {
        "soft": {
            "static_friction_coefficient": 0.5,
        },
        "rigid": {"K": 1e5},
        "relaxed": {"static_friction_coefficient": 1.0},
        "viscous": {
            "number_of_active_collidable_points_steady_state": 4,
            "static_friction_coefficient": 0.5,
            "damping_ratio": 1.0,
            "max_penetration": 0.001,
        },
    }

    rod_sdf = rod.Sdf(
        version="1.10",
        model=BoxBuilder(x=1, y=1, z=1, mass=1.0, name="box")
        .build_model()
        .add_link()
        .add_inertial()
        .add_visual()
        .add_collision()
        .build(),
    )

    model = js.model.JaxSimModel.build_from_model_description(
        model_description=rod_sdf,
        time_step=TIME_STEP,
        terrain=TERRAIN,
    )

    def change_contact_model(
        model: js.model.JaxSimModel, contact_model: str
    ) -> tuple[js.model.JaxSimModel, js.data.JaxSimModelData]:

        contact_model_cls = CONTACT_MODEL_DICT[contact_model]

        with model.editable(validate=False) as model:
            model.contact_model = contact_model_cls.build(terrain=TERRAIN)

        contact_model_params = js.contact.estimate_good_contact_parameters(
            model=model, **CONTACT_PARAMS_DICT[contact_model]
        )

        data = js.data.JaxSimModelData.build(
            model=model,
            base_position=jnp.array([0.0, 0.0, 1.0]),
            velocity_representation=VelRepr.Inertial,
            contacts_params=contact_model_params,
        )

        return model, data

    def run_simulation(
        model: js.model.JaxSimModel, contact_model: str
    ) -> tuple[list, list, js.data.JaxSimModelData]:

        model, data = change_contact_model(model=model, contact_model=contact_model)

        positions, orientations = [], []

        # Dummy step for compilation.
        if contact_model == "viscous":
            data, _ = jaxsim.rbda.contacts.visco_elastic.step(model=model, data=data)
        else:
            data, _ = js.model.step(
                model=model,
                data=data,
            )
        positions.append(data.base_position())
        orientations.append(data.base_orientation(dcm=False))
        iter_time = 0

        for _ in range(T):
            now = time.perf_counter()

            if contact_model == "viscous":
                data, _ = jaxsim.rbda.contacts.visco_elastic.step(
                    model=model, data=data
                )
            else:
                data, _ = js.model.step(
                    model=model,
                    data=data,
                )

            iter_time += time.perf_counter() - now

            positions.append(data.base_position())
            orientations.append(data.base_orientation(dcm=False))

        logging.info(
            f"Mean RTF: {TIME_STEP / (iter_time / (T + 1))}, Contact Model: {contact_model}"
        )

        return positions, orientations, data

    contact_model = "viscous"  # @param ["soft", "rigid", "relaxed", "viscous"]

    positions, orientations, data = run_simulation(
        model=model, contact_model=contact_model
    )

    mjcf_string, assets = jaxsim.mujoco.RodModelToMjcf.convert(
        rod_model=model.built_from.model,
        cameras=jaxsim.mujoco.loaders.MujocoCamera.build_from_target_view(
            camera_name="box_camera",
            lookat=js.link.com_position(
                model=model,
                data=data,
                link_index=0,
                in_link_frame=False,
            ),
            distance=3,
            azimut=150,
            elevation=-10,
        ),
        plane_normal=TERRAIN.normal(),
    )

    mj_model_helper = jaxsim.mujoco.MujocoModelHelper.build_from_xml(
        mjcf_description=mjcf_string, assets=assets
    )

    recorder = jaxsim.mujoco.MujocoVideoRecorder(
        model=mj_model_helper.model,
        data=mj_model_helper.data,
        fps=int(1 / model.time_step),
        width=320 * 2,
        height=320 * 2,
    )

    for position, orientation in zip(positions, orientations, strict=True):
        mj_model_helper.set_base_position(position=position)
        mj_model_helper.set_base_orientation(orientation=orientation)

        recorder.record_frame()

    recorder.write_video(path=Path.cwd() / f"{contact_model}_{TIME_STEP}.mp4")

    print(f"Finished recording {contact_model}_{TIME_STEP}")


if __name__ == "__main__":
    main()
