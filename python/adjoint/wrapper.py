"""Wrapper for converting a Meep simulation into a differentiable JAX callable function.

Usage example:
```
import jax.numpy as jnp
import meep as mp
import meep.adjoint as mpa

sources = [
  mp.EigenModeSource(...)
]

monitors = [
  mpa.EigenmodeCoefficient(...),
  mpa.EigenmodeCoefficient(...),
]

design_regions = [
  mpa.DesignRegion(...)
]

frequencies = [1/1.55, 1/1.60, 1/1.65, ...]

simulation = mp.Simulation(...)

wrapped_meep = MeepJaxWrapper(
    simulation,
    sources,
    monitors,
    design_regions,
    frequencies,
    measurement_interval = 50.0,
    dft_field_components = (mp.Ez,),
    dft_threshold = 1e-6,
    minimum_run_time = 0,
    maximum_run_time = onp.inf,
    until_after_sources = True
)

def loss(x):
    monitor_values = wrapped_meep([x])
    t = monitor_values[0,:] / monitor_values[1,:]
    # Mean transmission vs wavelength
    return jnp.mean(jnp.abs(t))

value, grad = jax.value_and_grad(loss)(x)
```
"""

from typing import Callable, List

import jax
import jax.numpy as jnp

from .wrapper_base import MeepWrapperBase


class MeepJaxWrapper(MeepWrapperBase):
    """Wraps a Meep simulation object into a JAX-differentiable callable.

    See `MeepWrapperBase` for the constructor arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._simulate_fn = self._initialize_callable()

    def __call__(self, designs: List[jnp.ndarray]) -> jnp.ndarray:
        """Performs a Meep simulation, taking a list of designs and returning mode overlaps.

        Args:
          designs: a list of design variables as 1D, 2D, or 3D JAX arrays. Valid shapes for
          design variables are (Nx, Ny, Nz) where Nx{y,z} match the elements of the
          `grid_size` constructor argument of Meep's `MaterialGrid` used for the
          corresponding design region. Singleton dimensions of the `grid_size` may be
          omitted from the corresponding design variable. For example, a design variable
          with a shape of either (10, 20) or (10, 20, 1) would be compatible with a
          `grid_size` of (10, 20, 1). Similarly, a design variable with shapes of (25,),
          (25, 1), or (25, 1, 1) would be compatible with a `grid_size` of (25, 1, 1).

        Returns:
          a complex-valued JAX ndarray of differentiable mode monitor overlaps values with
          a shape of (num monitors, num frequencies).
        """
        return self._simulate_fn(designs)

    def _initialize_callable(self) -> Callable[[List[jnp.ndarray]], jnp.ndarray]:
        """Initializes the callable JAX function and registers its VJP."""

        @jax.custom_vjp
        def simulate(design_variables: List[jnp.ndarray]) -> jnp.ndarray:
            monitor_values, _ = self._run_fwd_simulation(design_variables)
            return jnp.asarray(monitor_values)

        def _simulate_fwd(design_variables):
            """Runs forward simulation, returning monitor values and fields."""
            monitor_values, self.fwd_design_region_monitors = self._run_fwd_simulation(
                design_variables
            )
            design_variable_shapes = [x.shape for x in design_variables]
            return jnp.asarray(monitor_values), (design_variable_shapes)

        def _simulate_rev(res, monitor_values_grad):
            """Runs adjoint simulation, returning VJP of design wrt monitor values."""
            design_variable_shapes = res
            self.adj_design_region_monitors = self._run_adjoint_simulation(
                monitor_values_grad
            )
            vjps = self._calculate_vjps(
                self.fwd_design_region_monitors,
                self.adj_design_region_monitors,
                design_variable_shapes,
            )
            return ([jnp.asarray(vjp) for vjp in vjps],)

        simulate.defvjp(_simulate_fwd, _simulate_rev)

        return simulate
