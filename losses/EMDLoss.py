# implement


import torch
import emd_cuda


class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


def earth_mover_distance(xyz1, xyz2, transpose=True):
    """Earth Mover Distance (Approx)
    Args:
        xyz1 (torch.Tensor): (b, 3, n1)
        xyz2 (torch.Tensor): (b, 3, n1)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.
    Returns:
        cost (torch.Tensor): (b)
    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)
    return cost

if __name__ == '__main__':
    import torch
    import numpy as np
    import time

    # gt
    p1 = torch.from_numpy(np.array([[[1.7, -0.1, 0.1], [0.1, 1.2, 0.3]]], dtype=np.float32)).cuda()
    p1 = p1.repeat(3, 1, 1)
    p2 = torch.from_numpy(np.array([[[0.3, 1.8, 0.2], [1.2, -0.2, 0.3]]], dtype=np.float32)).cuda()
    p2 = p2.repeat(3, 1, 1)
    print(p1)
    print(p2)
    p1.requires_grad = True
    p2.requires_grad = True

    gt_dist = (((p1[0, 0] - p2[0, 1]) ** 2).sum() + ((p1[0, 1] - p2[0, 0]) ** 2).sum()) / 2 + \
              (((p1[1, 0] - p2[1, 1]) ** 2).sum() + ((p1[1, 1] - p2[1, 0]) ** 2).sum()) * 2 + \
              (((p1[2, 0] - p2[2, 1]) ** 2).sum() + ((p1[2, 1] - p2[2, 0]) ** 2).sum()) / 3
    print('gt_dist: ', gt_dist)

    gt_dist.backward()
    print(p1.grad)
    print(p2.grad)

    # emd
    p1 = torch.from_numpy(np.array([[[1.7, -0.1, 0.1], [0.1, 1.2, 0.3]]], dtype=np.float32)).cuda()
    p1 = p1.repeat(3, 1, 1)
    p2 = torch.from_numpy(np.array([[[0.3, 1.8, 0.2], [1.2, -0.2, 0.3]]], dtype=np.float32)).cuda()
    p2 = p2.repeat(3, 1, 1)
    print(p1.shape)
    print(p2.shape)
    p1.requires_grad = True
    p2.requires_grad = True

    d = earth_mover_distance(p1, p2, transpose=False)
    print("asdasd",d)

    loss = d[0] / 2 + d[1] * 2 + d[2] / 3
    print(loss)

    loss.backward()
    print(p1.grad)
    print(p2.grad)
