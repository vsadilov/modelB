import json
from pathlib import Path

import torch
import torch.nn as nn


def get_device():
    """Select the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def get_model_state_dict(model):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _strip_module_prefix(state_dict):
    return {
        k.replace('module.', '', 1) if k.startswith('module.') else k: v
        for k, v in state_dict.items()
    }


def _add_module_prefix(state_dict):
    return {k if k.startswith('module.') else f'module.{k}': v for k, v in state_dict.items()}


def _remap_legacy_keys(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        new_key = key.replace(".sa.", ".self_attn.")
        if new_key in remapped:
            remapped[key] = value
        else:
            remapped[new_key] = value
    return remapped


def load_model_state_dict(model, state_dict):
    if any(".sa." in k for k in state_dict.keys()):
        state_dict = _remap_legacy_keys(state_dict)
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    if isinstance(model, nn.DataParallel):
        if not has_module_prefix:
            state_dict = _add_module_prefix(state_dict)
    else:
        if has_module_prefix:
            state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict)


def move_optimizer_state(optimizer, device):
    """Move optimizer state tensors to the target device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _vocab_chars_path(save_path):
    return Path(save_path).with_suffix(".vocab_chars.json")


def save_vocab_chars(save_path, vocab_chars):
    if not vocab_chars:
        return
    path = _vocab_chars_path(save_path)
    path.write_text(json.dumps(vocab_chars, ensure_ascii=True), encoding="utf-8")


def load_vocab_chars(save_path):
    path = _vocab_chars_path(save_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
