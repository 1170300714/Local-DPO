import torch
from extensions.xfuser.core.distributed.group_coordinator import GroupCoordinator


class _SplitForwardGatherBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, dim, process_group: GroupCoordinator):
        ctx.process_group = process_group
        ctx.dim = dim
        output = torch.chunk(input_, process_group.world_size, dim=dim)[process_group.rank_in_group].clone()
        ctx.grad_scale = output.size(dim) / input_.size(dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx.grad_scale
        grad_output = grad_output.contiguous()
        grad_output = ctx.process_group.all_gather(grad_output, dim=ctx.dim)
        return grad_output, None, None


def split_forward_gather_backward(input_, dim, process_group):
    return _SplitForwardGatherBackward.apply(input_, dim, process_group)


class _GatherForwardSplitBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, dim, process_group: GroupCoordinator):
        input_ = input_.contiguous()
        ctx.process_group = process_group
        ctx.dim = dim
        output = process_group.all_gather(input_, dim=dim)
        ctx.grad_scale = output.size(dim) / input_.size(dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx.grad_scale
        pg = ctx.process_group
        grad_output = torch.chunk(grad_output, pg.world_size, dim=ctx.dim)[pg.rank_in_group].clone()
        return grad_output, None, None


def gather_forward_split_backward(input_, dim, process_group):
    return _GatherForwardSplitBackward.apply(input_, dim, process_group)
