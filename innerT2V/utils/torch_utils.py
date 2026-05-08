import torch
from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module


def get_gradient_norm(parameters):
    norm = 0
    for param in parameters:
        if param.grad is None:
            continue
        local_norm = param.grad.detach().data.norm(2)
        norm += local_norm.item() ** 2
    norm = norm**0.5
    return norm


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def summarize_model_info(model):
    def get_params_info(model):
        all_param = 0
        trainable_params = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        return all_param, trainable_params
    all_param, trainable_params = get_params_info(model)
    info_str = 'Params: {} ({:.3f} B), Learnable Params: {} ({:.3f} B, {:.2f}%)\n'.format(
        all_param, all_param / 1024 ** 3,
        trainable_params, trainable_params / 1024 ** 3,
        trainable_params / all_param * 100)
    info_str += '{}\n'.format(model.__repr__())
    return info_str
