from typing import Union
import unittest

import numpy as np


class ApproxComparisonTestCase(unittest.TestCase):
    """A mixin for adding correct scalar/vector comparison."""

    def assertClose(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        epsilon: float = 1e-2,
        msg: str = "",
    ):
        """Checks if two scalars or vectors satisfy ‖x-y‖ ≤ ε * max(‖x‖, ‖y‖).

        Args:
            x, y: two quantities to be compared (scalars or 1d arrays).
            epsilon: threshold value (maximum) of the relative error.
            msg: a string to display if the inequality is violated.
        """
        x = np.atleast_1d(x).ravel()
        y = np.atleast_1d(y).ravel()
        x_norm = np.linalg.norm(x, ord=np.inf)
        y_norm = np.linalg.norm(y, ord=np.inf)
        diff_norm = np.linalg.norm(x - y, ord=np.inf)
        self.assertLessEqual(diff_norm, epsilon * np.maximum(x_norm, y_norm), msg)


def build_straight_wg_simulation(
    wg_width=0.5,
    wg_padding=1.0,
    wg_length=1.0,
    pml_width=1.0,
    source_to_pml=0.5,
    source_to_monitor=0.1,
    frequencies=[1 / 1.55],
    gaussian_rel_width=0.2,
    sim_resolution=20,
    design_region_resolution=20,
):
    """Builds a simulation of a straight waveguide with a design region segment."""
    import meep as mp
    import meep.adjoint as mpa

    design_region_shape = (1.0, wg_width)

    # Simulation domain size
    sx = 2 * pml_width + 2 * wg_length + design_region_shape[0]
    sy = (
        2 * pml_width
        + 2 * wg_padding
        + max(
            wg_width,
            design_region_shape[1],
        )
    )

    # Mean / center frequency
    fmean = np.mean(frequencies)

    si = mp.Medium(index=3.4)
    sio2 = mp.Medium(index=1.44)

    sources = [
        mp.EigenModeSource(
            mp.GaussianSource(frequency=fmean, fwidth=fmean * gaussian_rel_width),
            eig_band=1,
            direction=mp.NO_DIRECTION,
            eig_kpoint=mp.Vector3(1, 0, 0),
            size=mp.Vector3(0, wg_width + 2 * wg_padding, 0),
            center=[-sx / 2 + pml_width + source_to_pml, 0, 0],
        ),
        mp.EigenModeSource(
            mp.GaussianSource(frequency=fmean, fwidth=fmean * gaussian_rel_width),
            eig_band=1,
            direction=mp.NO_DIRECTION,
            eig_kpoint=mp.Vector3(-1, 0, 0),
            size=mp.Vector3(0, wg_width + 2 * wg_padding, 0),
            center=[sx / 2 - pml_width - source_to_pml, 0, 0],
        ),
    ]
    nx, ny = int(design_region_shape[0] * design_region_resolution), int(
        design_region_shape[1] * design_region_resolution
    )
    mat_grid = mp.MaterialGrid(
        mp.Vector3(nx, ny),
        sio2,
        si,
        grid_type="U_DEFAULT",
    )

    design_regions = [
        mpa.DesignRegion(
            mat_grid,
            volume=mp.Volume(
                center=mp.Vector3(),
                size=mp.Vector3(
                    design_region_shape[0],
                    design_region_shape[1],
                    0,
                ),
            ),
        )
    ]

    geometry = [
        mp.Block(
            center=mp.Vector3(
                x=-design_region_shape[0] / 2 - wg_length / 2 - pml_width / 2
            ),
            material=si,
            size=mp.Vector3(wg_length + pml_width, wg_width, 0),
        ),  # left wg
        mp.Block(
            center=mp.Vector3(
                x=+design_region_shape[0] / 2 + wg_length / 2 + pml_width / 2
            ),
            material=si,
            size=mp.Vector3(wg_length + pml_width, wg_width, 0),
        ),  # right wg
        mp.Block(
            center=design_regions[0].center,
            size=design_regions[0].size,
            material=mat_grid,
        ),  # design region
    ]

    simulation = mp.Simulation(
        cell_size=mp.Vector3(sx, sy),
        boundary_layers=[mp.PML(pml_width)],
        geometry=geometry,
        sources=sources,
        resolution=sim_resolution,
    )

    monitor_centers = [
        mp.Vector3(-sx / 2 + pml_width + source_to_pml + source_to_monitor),
        mp.Vector3(sx / 2 - pml_width - source_to_pml - source_to_monitor),
    ]
    monitor_size = mp.Vector3(y=wg_width + 2 * wg_padding)

    monitors = [
        mpa.EigenmodeCoefficient(
            simulation,
            mp.Volume(center=center, size=monitor_size),
            mode=1,
            forward=forward,
        )
        for center in monitor_centers
        for forward in [True, False]
    ]
    return simulation, sources, monitors, design_regions, frequencies
