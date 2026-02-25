"""
Exponential Moving Average (EMA) for model parameters.
"""

from typing import Dict, Any, Optional
import torch


class EMAModel:
    """
    Exponential Moving Average (EMA) of model parameters.

    Maintains a shadow copy of model parameters that is updated as a moving average.
    Useful for stabilizing training and improving generalization.

    Features:
    - Keeps FP32 shadow copy on CPU by default (low GPU memory)
    - Can temporarily apply EMA weights to model for evaluation
    - Safe restore mechanism with try-finally protection

    Notes:
    - Only tracks parameters, not buffers (e.g., BatchNorm running stats)
    - Should track raw (non-DDP) module parameters

    Example:
        ema = EMAModel(model, decay=0.999)

        # Training loop
        for batch in loader:
            loss.backward()
            optimizer.step()
            ema.update(model)  # Update EMA after each step

        # Evaluation with EMA weights
        ema.apply_to(model)
        try:
            eval_loss = evaluate(model, val_loader)
        finally:
            ema.restore(model)  # Always restore!
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        *,
        device: str = "cpu",
        use_fp32: bool = True,
    ):
        """
        Initialize EMA.

        Args:
            model: Model to track
            decay: EMA decay rate (higher = more smoothing)
            device: Device to store shadow parameters ("cpu" recommended)
            use_fp32: Convert shadow parameters to FP32
        """
        self.decay = float(decay)
        self.device = torch.device(device)
        self.use_fp32 = bool(use_fp32)

        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Optional[Dict[str, torch.Tensor]] = None

        self._init_from(model)

    def _init_from(self, model: torch.nn.Module) -> None:
        """Initialize shadow parameters from model."""
        self.shadow.clear()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                t = p.detach()
                if self.use_fp32:
                    t = t.float()
                t = t.to(self.device, non_blocking=True).clone()
                self.shadow[name] = t

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """
        Update EMA parameters: shadow = decay*shadow + (1-decay)*param

        Call this after each optimizer step.
        """
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # Initialize if missing
            if name not in self.shadow:
                t = p.detach()
                if self.use_fp32:
                    t = t.float()
                self.shadow[name] = t.to(self.device, non_blocking=True).clone()
                continue

            # EMA update
            src = p.detach()
            if self.use_fp32:
                src = src.float()

            src = src.to(self.device, non_blocking=True)
            self.shadow[name].mul_(d).add_(src, alpha=(1.0 - d))

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        """
        Apply EMA weights to model, backing up current parameters.

        Must call restore() afterwards to recover original parameters.
        Use in try-finally block for safety:

            ema.apply_to(model)
            try:
                evaluate(model)
            finally:
                ema.restore(model)
        """
        self._backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = p.detach().clone()
                ema_t = self.shadow[name]
                if ema_t.dtype != p.dtype:
                    ema_t = ema_t.to(dtype=p.dtype)
                p.copy_(ema_t.to(device=p.device, non_blocking=True))

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        """
        Restore parameters backed up by apply_to().

        Safe to call multiple times or without prior apply_to().
        """
        if self._backup is None:
            return
        for name, p in model.named_parameters():
            if name in self._backup:
                p.copy_(self._backup[name].to(device=p.device, non_blocking=True))
        self._backup = None

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            "decay": self.decay,
            "device": str(self.device),
            "use_fp32": self.use_fp32,
            "shadow": {
                k: (v.cpu() if v.device.type != "cpu" else v)
                for k, v in self.shadow.items()
            },
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.decay = float(sd.get("decay", self.decay))
        self.device = torch.device(sd.get("device", str(self.device)))
        self.use_fp32 = bool(sd.get("use_fp32", self.use_fp32))

        shadow = sd.get("shadow", {})
        self.shadow = {}
        for k, v in shadow.items():
            if not torch.is_tensor(v):
                continue
            t = v
            if self.use_fp32:
                t = t.float()
            self.shadow[k] = t.to(self.device)

