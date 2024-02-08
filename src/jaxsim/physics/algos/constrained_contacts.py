import jax_dataclasses
import jax.numpy as jnp
import jaxopt
import numpy.typing as npt
from jaxsim.high_level.model import Model
from jaxsim.physics.terrain import FlatTerrain, Terrain


@jax_dataclasses.pytree_dataclass
class ConstrainedContactParams:
    """Parameters of the constrained contacts model."""

    timeconst: float = dataclasses.field(
        default_factory=lambda: jnp.array(0.1, dtype=float)
    )
    dampratio: float = dataclasses.field(
        default_factory=lambda: jnp.array(0.5, dtype=float)
    )
    dmin: float = dataclasses.field(default_factory=lambda: jnp.array(0.0, dtype=float))
    dmax: float = dataclasses.field(default_factory=lambda: jnp.array(1.0, dtype=float))
    width: float = dataclasses.field(
        default_factory=lambda: jnp.array(0.1, dtype=float)
    )
    mid: float = dataclasses.field(default_factory=lambda: jnp.array(0.5, dtype=float))
    power: float = dataclasses.field(
        default_factory=lambda: jnp.array(1.0, dtype=float)
    )

    @staticmethod
    def build(
        timeconst: float = 0.1,
        dampratio: float = 0.5,
        dmin: float = 0.0,
        dmax: float = 1.0,
        width: float = 0.1,
        mid: float = 0.5,
        power: float = 1.0,
    ) -> "ConstrainedContactParams":
        """
        Create a ConstrainedContactParams instance with specified parameters.

        Args:
            timeconst (float, optional): The time constant. Defaults to 0.1.
            dampratio (float, optional): The damping ratio. Defaults to 0.5.
            dmin (float, optional): The minimum damping. Defaults to 0.0.
            dmax (float, optional): The maximum damping. Defaults to 1.0.
            width (float, optional): The width of the damping function. Defaults to 0.1.
            mid (float, optional): The mid value of the damping function. Defaults to 0.5.
            power (float, optional): The power of the damping function. Defaults to 1.0.

        Returns:
            ConstrainedContactParams: A ConstrainedContactParams instance with the specified parameters.
        """

        return ConstrainedContactParams(
            timeconst=jnp.array(timeconst, dtype=float),
            dampratio=jnp.array(dampratio, dtype=float),
            dmin=jnp.array(dmin, dtype=float),
            dmax=jnp.array(dmax, dtype=float),
            width=jnp.array(width, dtype=float),
            mid=jnp.array(mid, dtype=float),
            power=jnp.array(power, dtype=float),
        )


@jax_dataclasses.pytree_dataclass
class ConstrainedContact:
    """Constrained contacts model."""

    parameters: SoftContactsParams = dataclasses.field(
        default_factory=ConstrainedContactParams
    )

    terrain: Terrain = dataclasses.field(default_factory=FlatTerrain)

    def contact_model(
        self,
        model: high_level.Model,
        position: jtp.Vector,
        velocity: jtp.Vector,
        tangential_deformation: jtp.Vector | None = None,
    ) -> Tuple[jtp.Vector, jtp.Vector]:
        """
        Compute the contact forces.

        Args:
            model (PhysicsModel): The physics model.
            position (jtp.Vector): The position of the collidable point.
            velocity (jtp.Vector): The linear velocity of the collidable point.
            tangential_deformation (jtp.Vector, optional): The tangential deformation. Defaults to None.

        Returns:
            Tuple[jtp.Vector, jtp.Vector]: A tuple containing the contact force and material deformation rate.
        """

        def _imp_aref(
            self: Self, position: jax.Array, velocity: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
            """Calculates impedance and offset acceleration in constraint frame.

            Args:
                params: solver params
                position: position in constraint frame
                velocity: velocity in constraint frame

            Returns:
                imp: constraint impedance
                Aref: offset acceleration in constraint frame
            """
            # this formulation corresponds to the parameterization described here:
            # https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
            timeconst, dampratio, dmin, dmax, width, mid, power = (
                self.parameters.timeconst,
                self.parameters.dampratio,
                self.parameters.dmin,
                self.parameters.dmax,
                self.parameters.width,
                self.parameters.mid,
                self.parameters.power,
            )

            impedance_x = jnp.abs(position) / width
            impedance_a = (1.0 / jnp.power(mid, power - 1)) * jnp.power(
                impedance_x, power
            )
            impedance_b = 1 - (1.0 / jnp.power(1 - mid, power - 1)) * jnp.power(
                1 - impedance_x, power
            )
            impedance_y = jnp.where(impedance_x < mid, impedance_a, impedance_b)
            impedance = dmin + impedance_y * (dmax - dmin)
            impedance = jnp.clip(impedance, dmin, dmax)
            impedance = jnp.where(impedance_x > 1.0, dmax, impedance)

            b = 2 / (dmax * timeconst)
            k = 1 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)

            # See https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
            stiffness, damping = params[:2]
            b = jnp.where(damping <= 0, -damping / dmax, b)
            k = jnp.where(stiffness <= 0, -stiffness / (dmax * dmax), k)

            aref = -b * velocity - k * impedance * position

            return impedance, aref

        def point_jacobian(
            sys: System,
            com: jax.Array,
            cdof: Motion,
            position: jax.Array,
            link_idx: jax.Array,
        ) -> Motion:
            """Calculates the jacobian of a point on a link.

            Args:
                sys: a brax system
                com: center of mass position
                cdof: dofs in com frame
                position: position in world frame to calculate the jacobian
                link_idx: index of link frame to transform point jacobian

            Returns:
                pt: point jacobian
            """

            # backward scan up tree: build the link mask corresponding to link_idx
            def mask_fn(mask_child, link):
                mask = link == link_idx
                if mask_child is not None:
                    mask += mask_child
                return mask

            mask = scan.tree(
                sys, mask_fn, "l", jnp.arange(sys.num_links()), reverse=True
            )
            cdof = jax.vmap(lambda a, b: a * b)(cdof, jnp.take(mask, sys.dof_link()))
            off = Transform.create(pos=position - com[link_idx])
            return off.vmap(in_axes=(None, 0)).do(cdof)

        def jac_limit(
            sys: System, state: State
        ) -> Tuple[jax.Array, jax.Array, jax.Array]:
            """Calculates the jacobian for angle limits in dof frame.

            Args:
                sys: a brax system
                state: generalized state

            Returns:
                jac: the angle limit jacobian
                pos: angle in constraint frame
                diag: approximate diagonal of A matrix
            """
            if sys.dof.limit is None:
                return jnp.zeros((0, sys.qd_size())), jnp.zeros((0,)), jnp.zeros((0,))

            # determine q and qd indices for non-free joints
            q_idx, qd_idx = sys.q_idx("123"), sys.qd_idx("123")

            pos_min = state.q[q_idx] - sys.dof.limit[0][qd_idx]
            pos_max = sys.dof.limit[1][qd_idx] - state.q[q_idx]
            pos = jnp.minimum(jnp.minimum(pos_min, pos_max), 0)

            side = ((pos_min < pos_max) * 2 - 1) * (pos < 0)
            jac = jax.vmap(jnp.multiply)(jnp.eye(sys.qd_size())[qd_idx], side)
            params = sys.dof.solver_params[qd_idx]
            imp, aref = jax.vmap(_imp_aref)(params, pos, jac @ state.qd)
            diag = sys.dof.invweight[qd_idx] * (pos < 0) * (1 - imp) / (imp + 1e-8)
            aref = jax.vmap(lambda x, y: x * y)(aref, (pos < 0))

            return jac, diag, aref

        def jac_contact(
            sys: System, state: State
        ) -> Tuple[jax.Array, jax.Array, jax.Array]:
            """Calculates the jacobian for contact constraints.

            Args:
                sys: the brax system
                state: generalized state

            Returns:
                jac: the contact jacobian
                pos: contact position in constraint frame
                diag: approximate diagonal of A matrix
            """
            c = contact.get(sys, state.x)

            if c is None:
                return jnp.zeros((0, sys.qd_size())), jnp.zeros((0,)), jnp.zeros((0,))

            def row_fn(c):
                link_a, link_b = c.link_idx
                a = point_jacobian(sys, state.root_com, state.cdof, c.pos, link_a)
                b = point_jacobian(sys, state.root_com, state.cdof, c.pos, link_b)
                diff = b.vel - a.vel

                # 4 pyramidal friction directions
                jac = []
                for d in -c.frame[1:]:
                    for f in [-c.friction[0], c.friction[0]]:
                        jac.append(diff @ (d * f + c.frame[0]))

                jac = jnp.stack(jac)
                pos = jnp.tile(c.dist, 4)
                solver_params = jnp.concatenate([c.solref, c.solimp])
                imp, aref = _imp_aref(solver_params, pos, jac @ state.qd)
                t = (
                    sys.link.invweight[link_a] * (link_a > -1)
                    + sys.link.invweight[link_b]
                )
                diag = jnp.tile(t + c.friction[0] * c.friction[0] * t, 4)
                diag *= 2 * c.friction[0] * c.friction[0] * (1 - imp) / (imp + 1e-8)

                return jax.tree_map(lambda x: x * (c.dist < 0), (jac, diag, aref))

            return jax.tree_map(jnp.concatenate, jax.vmap(row_fn)(c))

            jpds = jac_contact(sys, state), jac_limit(sys, state)

            jac, diag, aref = jax.tree_map(lambda *x: jnp.concatenate(x), *jpds)

            # Unpack the position of the collidable point
            px, py, pz = W_p_C = position.squeeze()
            vx, vy, vz = W_ṗ_C = velocity.squeeze()

            # Compute the terrain normal and the contact depth
            n̂ = self.terrain.normal(x=px, y=py).squeeze()
            h = jnp.array([0, 0, self.terrain.height(x=px, y=py) - pz])

            # Compute the penetration depth normal to the terrain
            δ = jnp.maximum(1e-4, jnp.dot(h, n̂))

            # Compute the penetration normal velocity
            δ̇ = -jnp.dot(W_ṗ_C, n̂)

            M_inv = jnp.linalg.inv(model.free_floating_mass_matrix(q=model.q))

            # Calculate quantities for the linear optimization problem
            A = jacobian @ M_inv @ jacobian.T
            b = jacobian @ M_inv @ qf_smooth - a_ref

            objective = lambda x: 0.5 * x.T @ A @ x + b @ x

            # Compute the 3D linear force in C[W] frame
            F = jaxopt.ProjectedGradient(objective).run(jnp.zeros_like(b))

            return F, jnp.zeros_like(F)
