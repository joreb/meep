"""Framework-agnostic base for wrapping a Meep simulation as a differentiable function.

Shared by `MeepJaxWrapper` (wrapper.py) and `MeepTorchWrapper` (torch_wrapper.py).
"""

from typing import Iterable, List, Tuple

import numpy as onp

import meep as mp

from . import DesignRegion, EigenmodeCoefficient, utils


class MeepWrapperBase:
    """Runs forward/adjoint Meep simulations and computes design VJPs.

    Attributes:
        simulation: the pre-configured Meep simulation object.
        sources: a list of Meep sources for the forward simulation.
        monitors: a list of eigenmode coefficient monitors from the `meep.adjoint`
          module.
        design_regions: a list of design regions from the `meep.adjoint` module.
        frequencies: the list of frequencies, in normalized Meep units.
        dft_threshold: the threshold for DFT field convergence. Once the norm of the
          change in the fields (the maximum over all design regions and field
          components) is less than this value, the simulation will be stopped.
        minimum_run_time: the minimum run time of the simulation, in Meep time
          units. The default value is 0.
        maximum_run_time: the maximum run time of the simulation, in Meep time
          units. The default value is infinity.
        until_after_sources: whether `maximum_run_time` should be ignored until the
          sources have turned off. This parameter specifies whether `until` or
          `until_after_sources` is used. See
          https://meep.readthedocs.io/en/latest/Python_User_Interface/#Simulation
          for more information. The default is true.
        finite_difference_step: step used for the finite-difference estimate of
          the permittivity derivative inside the gradient calculation.
    """

    _log_fn = print

    def __init__(
        self,
        simulation: mp.Simulation,
        sources: List[mp.Source],
        monitors: List[EigenmodeCoefficient],
        design_regions: List[DesignRegion],
        frequencies: List[float],
        dft_threshold: float = 1e-11,
        minimum_run_time: float = 0,
        maximum_run_time: float = onp.inf,
        until_after_sources: bool = True,
        finite_difference_step: float = utils.FD_DEFAULT,
    ):
        self.simulation = simulation
        self.sources = sources
        self.monitors = monitors
        self.design_regions = design_regions
        self.frequencies = frequencies
        self.dft_threshold = dft_threshold
        self.minimum_run_time = minimum_run_time
        self.maximum_run_time = maximum_run_time
        self.until_after_sources = until_after_sources
        self.finite_difference_step = finite_difference_step

    def _run_until_args(self) -> dict:
        return {
            (
                "until_after_sources" if self.until_after_sources else "until"
            ): mp.stop_when_dft_decayed(
                self.dft_threshold, self.minimum_run_time, self.maximum_run_time
            )
        }

    def _run_fwd_simulation(
        self, design_variables: Iterable[onp.ndarray]
    ) -> Tuple[onp.ndarray, List[List[mp.DftFields]]]:
        """Runs forward simulation, returning monitor values and design region fields."""
        utils.validate_and_update_design(self.design_regions, design_variables)
        self.simulation.reset_meep()
        self.simulation.change_sources(self.sources)
        utils.register_monitors(self.monitors, self.frequencies)
        fwd_design_region_monitors = utils.install_design_region_monitors(
            self.simulation,
            self.design_regions,
            self.frequencies,
        )
        self.simulation.init_sim()
        self.simulation.run(**self._run_until_args())

        monitor_values = utils.gather_monitor_values(self.monitors)
        return (monitor_values, fwd_design_region_monitors)

    def _run_adjoint_simulation(
        self, monitor_values_grad: onp.ndarray
    ) -> List[List[mp.DftFields]]:
        """Runs adjoint simulation, returning design region fields.

        `monitor_values_grad` follows the JAX/autograd cotangent convention, i.e.
        for a real loss L and complex monitor value s = u + iv it is dL/du - i dL/dv.
        """
        if not self.design_regions:
            raise RuntimeError(
                "An adjoint simulation was attempted when no design "
                "regions are present."
            )
        adjoint_sources = utils.create_adjoint_sources(
            self.monitors, monitor_values_grad
        )
        # TODO refactor with optimization_problem.py #
        self.simulation.restart_fields()
        self.simulation.clear_dft_monitors()
        self.simulation.change_sources(adjoint_sources)
        #                                            #
        adj_design_region_monitors = utils.install_design_region_monitors(
            self.simulation,
            self.design_regions,
            self.frequencies,
        )
        self.simulation.init_sim()
        self.simulation.run(**self._run_until_args())

        return adj_design_region_monitors

    def _calculate_vjps(
        self,
        fwd_fields: List[List[mp.DftFields]],
        adj_fields: List[List[mp.DftFields]],
        design_variable_shapes: List[Tuple[int, ...]],
        sum_freq_partials: bool = True,
    ) -> List[onp.ndarray]:
        """Calculates the VJP for a given set of forward and adjoint fields."""
        return utils.calculate_vjps(
            self.simulation,
            self.design_regions,
            self.frequencies,
            fwd_fields,
            adj_fields,
            design_variable_shapes,
            sum_freq_partials=sum_freq_partials,
            finite_difference_step=self.finite_difference_step,
        )
