import unittest

import jax
import jax.numpy as jnp
import meep.adjoint as mpa
import numpy as onp
import parameterized
from utils import ApproxComparisonTestCase, build_straight_wg_simulation

import meep as mp

# The calculation of finite-difference gradients
# requires that JAX be operated with double precision
jax.config.update("jax_enable_x64", True)

# The step size for the finite-difference
# gradient calculation
_FD_STEP = 1e-4

# The tolerance for the adjoint and finite-difference
# gradient comparison
_TOL = 0.1 if mp.is_single_precision() else 0.025

# We expect 3 design region monitor pointers
# (one for each field component)
_NUM_DES_REG_MON = 3

mp.verbosity(0)


class UtilsTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        (
            self.simulation,
            self.sources,
            self.monitors,
            self.design_regions,
            self.frequencies,
        ) = build_straight_wg_simulation()

    def test_mode_monitor_helpers(self):
        mpa.utils.register_monitors(self.monitors, self.frequencies)
        self.simulation.run(until=100)
        monitor_values = mpa.utils.gather_monitor_values(self.monitors)
        self.assertEqual(monitor_values.dtype, onp.complex128)
        self.assertEqual(
            monitor_values.shape, (len(self.monitors), len(self.frequencies))
        )

    def test_dist_dft_pointers(self):
        fwd_design_region_monitors = mpa.utils.install_design_region_monitors(
            self.simulation,
            self.design_regions,
            self.frequencies,
        )
        self.assertEqual(len(fwd_design_region_monitors[0]), _NUM_DES_REG_MON)


class WrapperTest(ApproxComparisonTestCase):
    @parameterized.parameterized.expand(
        [
            (
                "1500_1550bw_01relative_gaussian_port1",
                onp.linspace(1 / 1.50, 1 / 1.55, 3).tolist(),
                0.1,
                0.5,
                0,
            ),
            (
                "1550_1600bw_02relative_gaussian_port1",
                onp.linspace(1 / 1.55, 1 / 1.60, 3).tolist(),
                0.2,
                0.5,
                0,
            ),
            (
                "1500_1600bw_03relative_gaussian_port1",
                onp.linspace(1 / 1.50, 1 / 1.60, 4).tolist(),
                0.3,
                0.5,
                0,
            ),
            (
                "1500_1550bw_01relative_gaussian_port2",
                onp.linspace(1 / 1.50, 1 / 1.55, 3).tolist(),
                0.1,
                0.5,
                1,
            ),
            (
                "1550_1600bw_02relative_gaussian_port2",
                onp.linspace(1 / 1.55, 1 / 1.60, 3).tolist(),
                0.2,
                0.5,
                1,
            ),
            (
                "1500_1600bw_03relative_gaussian_port2",
                onp.linspace(1 / 1.50, 1 / 1.60, 4).tolist(),
                0.3,
                0.5,
                1,
            ),
        ]
    )
    def test_wrapper_gradients(
        self,
        _,
        frequencies,
        gaussian_rel_width,
        design_variable_fill_value,
        excite_port_idx,
    ):
        """Tests gradient from the JAX-Meep wrapper against finite differences."""
        (
            simulation,
            sources,
            monitors,
            design_regions,
            frequencies,
        ) = build_straight_wg_simulation(
            frequencies=frequencies, gaussian_rel_width=gaussian_rel_width
        )

        design_shape = tuple(
            int(i) for i in design_regions[0].design_parameters.grid_size
        )[:2]
        x = onp.ones(design_shape) * design_variable_fill_value

        # Define a loss function
        def loss_fn(x, excite_port_idx=0):
            wrapped_meep = mpa.MeepJaxWrapper(
                simulation,
                [sources[excite_port_idx]],
                monitors,
                design_regions,
                frequencies,
            )
            monitor_values = wrapped_meep([x])
            s1p, s1m, s2p, s2m = monitor_values
            t = s2p / s1p if excite_port_idx == 0 else s1m / s2m
            return jnp.mean(jnp.square(jnp.abs(t)))

        value, adjoint_grad = jax.value_and_grad(loss_fn)(
            x, excite_port_idx=excite_port_idx
        )

        projection = []
        fd_projection = []

        # Project along 5 random directions in the design parameter space.
        for seed in range(5):
            # Create dp
            random_perturbation_vector = _FD_STEP * jax.random.normal(
                jax.random.PRNGKey(seed),
                x.shape,
            )

            # Calculate p + dp
            x_perturbed = x + random_perturbation_vector

            # Calculate T(p + dp)
            value_perturbed = loss_fn(x_perturbed, excite_port_idx=excite_port_idx)

            projection.append(
                onp.dot(
                    random_perturbation_vector.ravel(),
                    adjoint_grad.ravel(),
                )
            )
            fd_projection.append(value_perturbed - value)

        projection = onp.stack(projection)
        fd_projection = onp.stack(fd_projection)

        # Check that dp . ∇T ~ T(p + dp) - T(p)
        self.assertClose(
            projection,
            fd_projection,
            epsilon=_TOL,
        )


if __name__ == "__main__":
    unittest.main()
