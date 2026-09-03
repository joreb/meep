import unittest

import meep.adjoint as mpa
import numpy as onp
import parameterized
import torch
from utils import ApproxComparisonTestCase, build_straight_wg_simulation

import meep as mp

# The step size for the finite-difference gradient calculation
_FD_STEP = 1e-4

# The tolerance for the adjoint and finite-difference gradient comparison
_TOL = 0.1 if mp.is_single_precision() else 0.025

mp.verbosity(0)


class WrapperTest(ApproxComparisonTestCase):
    @parameterized.parameterized.expand(
        [
            (
                "1500_1550bw_01relative_gaussian_port1",
                onp.linspace(1 / 1.50, 1 / 1.55, 3).tolist(),
                0.1,
                0.5,
                0,
                False,
            ),
            (
                "1550_1600bw_02relative_gaussian_port2_checkpoint",
                onp.linspace(1 / 1.55, 1 / 1.60, 3).tolist(),
                0.2,
                0.5,
                1,
                True,
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
        checkpoint,
    ):
        """Tests gradient from the PyTorch-Meep wrapper against finite differences."""
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
        x = torch.full(design_shape, design_variable_fill_value, dtype=torch.float64)

        wrapped_meep = mpa.MeepTorchWrapper(
            simulation,
            [sources[excite_port_idx]],
            monitors,
            design_regions,
            frequencies,
            checkpoint=checkpoint,
        )

        def loss_fn(x):
            monitor_values = wrapped_meep([x])
            s1p, s1m, s2p, s2m = monitor_values
            t = s2p / s1p if excite_port_idx == 0 else s1m / s2m
            return torch.mean(torch.square(torch.abs(t)))

        x.requires_grad_(True)
        value = loss_fn(x)
        if checkpoint:
            # Forward fields must have been released before backward.
            self.assertIsNone(simulation.fields)
        value.backward()
        adjoint_grad = x.grad.numpy()
        value = value.item()
        x = x.detach()

        projection = []
        fd_projection = []

        # Project along 5 random directions in the design parameter space.
        for seed in range(5):
            random_perturbation_vector = _FD_STEP * torch.randn(
                x.shape, generator=torch.Generator().manual_seed(seed), dtype=x.dtype
            )
            x_perturbed = x + random_perturbation_vector
            with torch.no_grad():
                value_perturbed = loss_fn(x_perturbed).item()

            projection.append(
                onp.dot(
                    random_perturbation_vector.numpy().ravel(), adjoint_grad.ravel()
                )
            )
            fd_projection.append(value_perturbed - value)

        projection = onp.stack(projection)
        fd_projection = onp.stack(fd_projection)

        # Check that dp . ∇T ~ T(p + dp) - T(p)
        self.assertClose(projection, fd_projection, epsilon=_TOL)


if __name__ == "__main__":
    unittest.main()
