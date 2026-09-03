"""Wrapper for converting a Meep simulation into a differentiable PyTorch function.

Usage example:
```
import torch
import meep as mp
import meep.adjoint as mpa

wrapped_meep = mpa.MeepTorchWrapper(
    simulation,
    sources,
    monitors,
    design_regions,
    frequencies,
    checkpoint=True,  # recompute forward fields in backward instead of holding them
)

x = torch.rand(design_shape, dtype=torch.float64, device="cuda", requires_grad=True)

def loss(x):
    monitor_values = wrapped_meep([x])  # complex tensor, same device as x
    t = monitor_values[0, :] / monitor_values[1, :]
    return torch.mean(torch.abs(t))

loss(x).backward()
x.grad  # on "cuda"
```

Only the design arrays cross the host boundary: Meep's FDTD kernel runs on the
CPU, everything before and after it (filters, projections, loss) stays on
whatever device the design tensors live on.
"""

from typing import List

import numpy as onp
import torch

import meep as mp

from .wrapper_base import MeepWrapperBase


class _MeepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, wrapper, *designs):
        design_np = [d.detach().cpu().numpy() for d in designs]
        monitor_values, fwd_monitors = wrapper._run_fwd_simulation(design_np)
        ctx.wrapper = wrapper
        ctx.design_shapes = [d.shape for d in designs]
        ctx.design_meta = [(d.device, d.dtype) for d in designs]
        if wrapper.checkpoint:
            # Keep only the (small) design tensors; forward DFT fields are
            # recomputed in backward().
            ctx.save_for_backward(*designs)
            wrapper._free_fields()
        else:
            wrapper.fwd_design_region_monitors = fwd_monitors
        d0 = designs[0]
        complex_dtype = (
            torch.complex64 if d0.dtype == torch.float32 else torch.complex128
        )
        return torch.as_tensor(monitor_values, dtype=complex_dtype, device=d0.device)

    @staticmethod
    def backward(ctx, grad_output):
        wrapper = ctx.wrapper
        if wrapper.checkpoint:
            design_np = [d.detach().cpu().numpy() for d in ctx.saved_tensors]
            _, wrapper.fwd_design_region_monitors = wrapper._run_fwd_simulation(
                design_np
            )
        # PyTorch's cotangent for a real loss L wrt complex s=u+iv is dL/du + i dL/dv,
        # the conjugate of the JAX/autograd convention the adjoint machinery expects.
        monitor_values_grad = onp.conj(grad_output.detach().cpu().numpy())
        wrapper.adj_design_region_monitors = wrapper._run_adjoint_simulation(
            monitor_values_grad
        )
        vjps = wrapper._calculate_vjps(
            wrapper.fwd_design_region_monitors,
            wrapper.adj_design_region_monitors,
            ctx.design_shapes,
        )
        if wrapper.checkpoint:
            wrapper._free_fields()
        grads = [
            torch.as_tensor(vjp, dtype=dtype, device=device)
            for vjp, (device, dtype) in zip(vjps, ctx.design_meta)
        ]
        return (None, *grads)


class MeepTorchWrapper(MeepWrapperBase):
    """Wraps a Meep simulation object into a PyTorch-differentiable callable.

    See `MeepWrapperBase` for the common constructor arguments.

    Additional attributes:
        checkpoint: if true, the forward design-region DFT fields are not kept
          between the forward and backward passes; instead the forward simulation
          is re-run inside `backward()` (gradient checkpointing). This trades one
          extra forward FDTD run per backward for freeing all field memory as soon
          as the forward pass returns, which prevents out-of-memory failures when
          many wrapped simulations (or many calls of the same one) contribute to a
          single loss. It also allows several wrappers to share one `mp.Simulation`.
    """

    def __init__(self, *args, checkpoint: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint = checkpoint
        self.fwd_design_region_monitors = None
        self.adj_design_region_monitors = None

    def _free_fields(self) -> None:
        """Drops all field/DFT data held by the simulation."""
        self.fwd_design_region_monitors = None
        self.adj_design_region_monitors = None
        self.simulation.reset_meep()

    def __call__(self, designs: List[torch.Tensor]) -> torch.Tensor:
        """Performs a Meep simulation, taking a list of designs and returning mode overlaps.

        Args:
          designs: a list of design variables as 1D, 2D, or 3D real tensors on any
          device. See `MeepJaxWrapper.__call__` for the shape rules.

        Returns:
          a complex tensor of mode monitor overlaps with shape
          (num monitors, num frequencies), on the device of `designs[0]`.
        """
        return _MeepFunction.apply(self, *designs)
