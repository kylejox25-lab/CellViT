"""Apply one accumulated update, allowing GradScaler to recover from overflow."""

import torch


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    gradient_multiplier: float = 1.0,
) -> tuple[bool, float | None]:
    """Return (updated, pre-clipping norm); a skipped AMP update has no norm.

    unscale_ records non-finite gradients for scaler.step. Do not clip those
    gradients: let step skip the update and update lower the loss scale.
    The caller clears gradients and records whether parameters were updated.
    """
    scaler.unscale_(optimizer)
    parameters = [
        p for group in optimizer.param_groups for p in group["params"]
        if p.grad is not None
    ]
    if not parameters:
        raise RuntimeError("No gradients were produced")
    finite = torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()
    if not bool(finite):
        if not scaler.is_enabled():
            raise FloatingPointError("Non-finite gradients with AMP disabled; inspect loss and inputs")
        scaler.step(optimizer)  # Skipped using the overflow recorded by unscale_.
        scaler.update()
        return False, None

    for parameter in parameters:
        parameter.grad.mul_(gradient_multiplier)
    # Keep the guard for overflow in norm calculation or gradient correction.
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return True, float(norm)
