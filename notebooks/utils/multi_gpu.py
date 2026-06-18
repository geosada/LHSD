# Robust DataParallel shim for LightweightTrainer-style loops.
# - Moves the entire module tree to device_ids[0] (e.g., cuda:0)
# - Guards FUTURE register_buffer / register_parameter / add_module so that anything
#   created lazily inside perturb_batch / loss / forward also lands on cuda:0
# - Aliases forward -> loss for DP
# - Reduces outputs to a scalar loss for backward()
# - Includes optional verbose logs to pinpoint any late CPU creations

from __future__ import annotations
import torch
from torch import nn
from torch.nn.parallel import DataParallel
from typing import Any, Iterable, Optional, Sequence

__all__ = ["wrap_dataparallel_for_trainer", "build_dp_loader_from_tensor"]

# ---------------------------
# Public helper: CPU DataLoader
# ---------------------------
def build_dp_loader_from_tensor(
    tensor: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        tensor, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

# ---------------------------
# Internal utilities
# ---------------------------
def _alias_forward_to_loss_instance(model: nn.Module) -> None:
    if not hasattr(model, "loss"):
        raise AttributeError("model must define .loss(batch)")
    def _forward(batch):
        return model.loss(batch)
    setattr(model, "forward", _forward)

def _reduce_to_scalar(out: Any) -> torch.Tensor:
    # Accept dict/list/tuple/scalar and return a 0D tensor on CUDA if available
    if isinstance(out, dict):
        out = out.get("loss", next((v for v in out.values() if torch.is_tensor(v)), out))
    if isinstance(out, (list, tuple)):
        out = out[0]
    if not torch.is_tensor(out):
        dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        out = torch.as_tensor(out, device=dev)
    if out.dim() > 0:
        out = out.mean()
    return out

def _attach_device_guard_hooks(root: nn.Module):
    # Best-effort guard: if a leaf module has a weight, auto-move inputs to its device
    def _pre_hook(mod: nn.Module, inputs):
        dev = None
        if hasattr(mod, "weight") and torch.is_tensor(mod.weight):
            dev = mod.weight.device
        elif hasattr(mod, "bias") and torch.is_tensor(mod.bias):
            dev = mod.bias.device
        if dev is None:
            return inputs
        moved = []
        for x in inputs:
            if torch.is_tensor(x) and x.device != dev:
                moved.append(x.to(dev))
            elif isinstance(x, (list, tuple)):
                moved.append(type(x)(xi.to(dev) if torch.is_tensor(xi) and xi.device != dev else xi for xi in x))
            elif isinstance(x, dict):
                moved.append({k: (v.to(dev) if torch.is_tensor(v) and v.device != dev else v) for k, v in x.items()})
            else:
                moved.append(x)
        return tuple(moved)

    for m in root.modules():
        has_leaf_params = any(True for _ in m.parameters(recurse=False))
        if has_leaf_params:
            m.register_forward_pre_hook(_pre_hook, with_kwargs=False)

def _move_existing_buffers_to(device: torch.device, root: nn.Module):
    for m in root.modules():
        for name, buf in list(m.named_buffers(recurse=False)):
            if torch.is_tensor(buf) and buf.device != device:
                with torch.no_grad():
                    m._buffers[name] = buf.to(device)

def _move_existing_params_to(device: torch.device, root: nn.Module):
    for m in root.modules():
        for name, p in list(m.named_parameters(recurse=False)):
            if torch.is_tensor(p) and p.device != device:
                with torch.no_grad():
                    m._parameters[name] = nn.Parameter(p.to(device), requires_grad=p.requires_grad)

def _force_future_buffers_to(device: torch.device, root: nn.Module, *, verbose: bool = True):
    # Monkeypatch register_buffer so any *later* buffers land on the target device
    for m in root.modules():
        orig = m.register_buffer
        def rb(name, tensor, *args, _orig=orig, _dev=device, _m=m, **kwargs):
            dev_before = (str(tensor.device) if torch.is_tensor(tensor) else "N/A")
            if torch.is_tensor(tensor) and tensor.device != _dev:
                tensor = tensor.to(_dev)
            if verbose:
                print(f"[DPGuard] register_buffer: {type(_m).__name__}.{name} moved {dev_before} -> {_dev}")
            return _orig(name, tensor, *args, **kwargs)
        m.register_buffer = rb  # type: ignore[method-assign]

def _force_future_parameters_to(device: torch.device, root: nn.Module, *, verbose: bool = True):
    # Monkeypatch register_parameter so any *later* parameters land on the target device
    for m in root.modules():
        orig = m.register_parameter
        def rp(name, param, _orig=orig, _dev=device, _m=m):
            dev_before = (str(param.device) if isinstance(param, nn.Parameter) else "N/A")
            if isinstance(param, nn.Parameter) and param.device != _dev:
                param = nn.Parameter(param.to(_dev), requires_grad=param.requires_grad)
            if verbose:
                print(f"[DPGuard] register_parameter: {type(_m).__name__}.{name} moved {dev_before} -> {_dev}")
            return _orig(name, param)
        m.register_parameter = rp  # type: ignore[method-assign]

def _force_future_modules_to(device: torch.device, root: nn.Module, *, verbose: bool = True):
    # Monkeypatch add_module so any *later* submodules are moved to device and recursively protected
    for m in root.modules():
        orig = m.add_module
        def am(name, module, _orig=orig, _dev=device, _parent=m):
            if isinstance(module, nn.Module):
                module = module.to(_dev)
                # Recursively protect the new child
                _move_existing_params_to(_dev, module)
                _move_existing_buffers_to(_dev, module)
                _force_future_parameters_to(_dev, module, verbose=verbose)
                _force_future_buffers_to(_dev, module, verbose=verbose)
                _force_future_modules_to(_dev, module, verbose=verbose)
                if verbose:
                    print(f"[DPGuard] add_module: {type(_parent).__name__}.{name} -> {type(module).__name__} on {_dev}")
            return _orig(name, module)
        m.add_module = am  # type: ignore[method-assign]

def _debug_first_off_device(root: nn.Module, expect: str = "cuda:0"):
    dev = torch.device(expect)
    for n, p in root.named_parameters():
        if p.device != dev:
            print(f"[DPGuard][PARAM-CPU] {n} is on {p.device}, expected {dev}")
            return
    for n, b in root.named_buffers():
        if b.device != dev:
            print(f"[DPGuard][BUFFER-CPU] {n} is on {b.device}, expected {dev}")
            return

# ---------------------------
# The DP wrapper class
# ---------------------------
class _DPShim(nn.Module):
    """
    Wrap a training module for torch.nn.DataParallel while keeping a
    LightweightTrainer-like API (loss, forward, configure_optimizers, unpack_batch).

    Key behaviors:
      - Moves the entire module to cuda:{device_ids[0]}.
      - Ensures any parameters/buffers/submodules created *after* wrapping
        are also placed on the same device.
      - Aliases forward -> loss so DP can call it.
      - Reduces outputs to a scalar for backward().
    """
    def __init__(self, module: nn.Module, device_ids: Optional[Sequence[int]] = None, verbose: bool = True):
        super().__init__()
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError("DataParallel requires at least one visible CUDA device.")

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if len(device_ids) == 0:
            raise RuntimeError("No CUDA devices specified for DataParallel.")
        src = torch.device(f"cuda:{device_ids[0]}")

        # 1) Move the whole tree to the source device
        module = module.to(src)

        # 2) Fix any pre-existing CPU params/buffers (paranoid)
        _move_existing_params_to(src, module)
        _move_existing_buffers_to(src, module)

        # 3) Guard future creations (buffers, params, and submodules)
        _force_future_parameters_to(src, module, verbose=verbose)
        _force_future_buffers_to(src, module, verbose=verbose)
        _force_future_modules_to(src, module, verbose=verbose)

        # 4) Alias forward->loss and attach input device guards
        _alias_forward_to_loss_instance(module)
        _attach_device_guard_hooks(module)

        # 5) Wrap in DP; keep wrapper itself on src (optional but tidy)
        self.dp = DataParallel(module, device_ids=device_ids).cuda(src)
        self._src = src
        self._verbose = verbose

    # ---- Trainer entrypoints ----
    def loss(self, batch):
        _move_existing_params_to(self._src, self.dp.module)
        _move_existing_buffers_to(self._src, self.dp.module)

        # sanity ping (will print a culprit if any remain)
        _debug_first_off_device(self.dp.module, expect=str(self._src))

        return _reduce_to_scalar(self.dp(batch))

    def forward(self, batch):
        # not used by your trainer, but kept consistent
        return self.loss(batch)

    # ---- Passthroughs / conveniences ----
    def configure_optimizers(self):
        # Some trainers expect .module inside DP; expose it the same way
        if hasattr(self.dp.module, "configure_optimizers"):
            return self.dp.module.configure_optimizers()
        raise AttributeError("Wrapped module has no configure_optimizers().")

    def unpack_batch(self, batch):
        if hasattr(self.dp.module, "unpack_batch"):
            batch = self.dp.module.unpack_batch(batch)
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if not torch.is_tensor(batch):
            batch = torch.as_tensor(batch)
        return batch

    def __getattr__(self, name):
        # Delegate unknown attrs to the wrapped module
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.dp.module, name)

    def state_dict(self, *a, **k):
        return self.dp.module.state_dict(*a, **k)

    def load_state_dict(self, *a, **k):
        return self.dp.module.load_state_dict(*a, **k)

    def parameters(self, *a, **k):
        return self.dp.module.parameters(*a, **k)

# ---------------------------
# Public factory
# ---------------------------
def wrap_dataparallel_for_trainer(
    model: nn.Module,
    device_ids: Optional[Sequence[int]] = None,
    *,
    verbose: bool = True,
) -> nn.Module:
    """
    Wrap a model with a DP shim usable by LightweightTrainer.
    Example:
        training_module = wrap_dataparallel_for_trainer(training_module)
    """
    return _DPShim(model, device_ids=device_ids, verbose=verbose)

