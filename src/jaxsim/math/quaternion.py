import jax.lax
import jax.numpy as jnp
import jaxlie

import jaxsim.typing as jtp

from .utils import safe_norm


class Quaternion:
    """
    A utility class for quaternion operations.
    """

    @staticmethod
    def to_xyzw(wxyz: jtp.Vector) -> jtp.Vector:
        """
        Convert a quaternion from WXYZ to XYZW representation.

        Args:
            wxyz: Quaternion in WXYZ representation.

        Returns:
            Quaternion in XYZW representation.
        """
        return wxyz.squeeze()[jnp.array([1, 2, 3, 0])]

    @staticmethod
    def to_wxyz(xyzw: jtp.Vector) -> jtp.Vector:
        """
        Convert a quaternion from XYZW to WXYZ representation.

        Args:
            xyzw: Quaternion in XYZW representation.

        Returns:
            Quaternion in WXYZ representation.
        """
        return xyzw.squeeze()[jnp.array([3, 0, 1, 2])]

    @staticmethod
    def to_dcm(quaternion: jtp.Vector) -> jtp.Matrix:
        """
        Convert a quaternion to a direction cosine matrix (DCM).

        Args:
            quaternion: Quaternion in XYZW representation.

        Returns:
            The Direction cosine matrix (DCM).
        """
        return jaxlie.SO3(wxyz=quaternion).as_matrix()

    @staticmethod
    def from_dcm(dcm: jtp.Matrix) -> jtp.Vector:
        """
        Convert a direction cosine matrix (DCM) to a quaternion.

        Args:
            dcm: Direction cosine matrix (DCM).

        Returns:
            Quaternion in WXYZ representation.
        """
        return jaxlie.SO3.from_matrix(matrix=dcm).wxyz

    @staticmethod
    def derivative(
        quaternion: jtp.Vector,
        omega: jtp.Vector,
        omega_in_body_fixed: bool = False,
        K: float = 0.1,
    ) -> jtp.Vector:
        """
        Compute the derivative of a quaternion given angular velocity.

        Args:
            quaternion: Quaternion in XYZW representation.
            omega: Angular velocity vector.
            omega_in_body_fixed (bool): Whether the angular velocity is in the body-fixed frame.
            K (float): A scaling factor.

        Returns:
            The derivative of the quaternion.
        """
        ω = omega.squeeze()
        quaternion = quaternion.squeeze()

        def Q_body(q: jtp.Vector) -> jtp.Matrix:
            qw, qx, qy, qz = q

            return jnp.array(
                [
                    [qw, -qx, -qy, -qz],
                    [qx, qw, -qz, qy],
                    [qy, qz, qw, -qx],
                    [qz, -qy, qx, qw],
                ]
            )

        def Q_inertial(q: jtp.Vector) -> jtp.Matrix:
            qw, qx, qy, qz = q

            return jnp.array(
                [
                    [qw, -qx, -qy, -qz],
                    [qx, qw, qz, -qy],
                    [qy, -qz, qw, qx],
                    [qz, qy, -qx, qw],
                ]
            )

        Q = jax.lax.cond(
            pred=omega_in_body_fixed,
            true_fun=Q_body,
            false_fun=Q_inertial,
            operand=quaternion,
        )

        norm_ω = safe_norm(ω)

        qd = 0.5 * (
            Q
            @ jnp.hstack(
                [
                    K * norm_ω * (1 - safe_norm(quaternion)),
                    ω,
                ]
            )
        )

        return jnp.vstack(qd)

    @staticmethod
    def integration(
        quaternion: jtp.VectorLike,
        dt: jtp.FloatLike,
        omega: jtp.VectorLike,
        omega_in_body_fixed: jtp.BoolLike = False,
    ) -> jtp.Vector:
        """
        Integrate a quaternion in SO(3) given an angular velocity.

        Args:
            quaternion: The quaternion to integrate.
            dt: The time step.
            omega: The angular velocity vector.
            omega_in_body_fixed:
                Whether the angular velocity is in body-fixed representation
                as opposed to the default inertial-fixed representation.

        Returns:
            The integrated quaternion.
        """

        ω_AB = jnp.array(omega).squeeze().astype(float)
        A_Q_B = jnp.array(quaternion).squeeze().astype(float)

        # Build the initial SO(3) quaternion.
        W_Q_B_t0 = jaxlie.SO3(wxyz=A_Q_B)

        # Integrate the quaternion on the manifold.
        W_Q_B_tf = jax.lax.select(
            pred=omega_in_body_fixed,
            on_true=(W_Q_B_t0 @ jaxlie.SO3.exp(tangent=dt * ω_AB)).wxyz,
            on_false=(jaxlie.SO3.exp(tangent=dt * ω_AB) @ W_Q_B_t0).wxyz,
        )

        return W_Q_B_tf
