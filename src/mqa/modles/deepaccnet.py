"""Configurable DeepAccNet and protein-representation fusion model."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .resnet import LegacyResNet


def scatter_voxels(
    indices: torch.Tensor,
    values: torch.Tensor,
    shape: tuple[int, int, int, int, int],
) -> torch.Tensor:
    if indices.ndim != 2 or indices.shape[1] != len(shape):
        raise ValueError(f"Voxel indices must be [N, {len(shape)}], got {tuple(indices.shape)}")
    strides = torch.tensor(
        [math.prod(shape[index + 1 :]) for index in range(len(shape))],
        device=indices.device,
        dtype=torch.long,
    )
    flat_indices = (indices.long() * strides).sum(dim=1)
    output = values.new_zeros(math.prod(shape))
    output.scatter_add_(0, flat_indices, values)
    return output.view(shape)


def calculate_lddt(estogram: torch.Tensor, mask: torch.Tensor, center: int = 7) -> torch.Tensor:
    """Calculate residue-level lDDT from a 15-bin estogram and pair mask."""
    if estogram.ndim != 3 or mask.ndim != 2:
        raise ValueError("Expected estogram [15, L, L] and mask [L, L]")
    length = mask.shape[-1]
    off_diagonal = 1.0 - torch.eye(length, device=mask.device, dtype=mask.dtype)
    mask = mask * off_diagonal
    weighted = estogram * mask.unsqueeze(0)
    p0 = weighted[center].sum(dim=0)
    p1 = (weighted[center - 1] + weighted[center + 1]).sum(dim=0)
    p2 = (weighted[center - 2] + weighted[center + 2]).sum(dim=0)
    p3 = (weighted[center - 3] + weighted[center + 3]).sum(dim=0)
    denominator = mask.sum(dim=0).clamp_min(1e-8)
    return 0.25 * (4.0 * p0 + 3.0 * p1 + 2.0 * p2 + p3) / denominator


class DeepAccNetFusion(nn.Module):
    """DeepAccNet with a learned per-residue representation projection."""

    def __init__(
        self,
        representation_dim: int,
        representation_hidden_dims: Sequence[int],
        *,
        one_body_dim: int = 70,
        two_body_dim: int = 33,
        projection_dim: int = 64,
        chunks: int = 5,
        channels: int = 128,
        residue_types: int = 20,
        projection_name: str = "esm_mlp",
    ) -> None:
        super().__init__()
        self.representation_dim = representation_dim
        self.one_body_dim = one_body_dim
        self.two_body_dim = two_body_dim
        self.residue_types = residue_types

        self.retype = nn.Conv3d(residue_types, 20, 1, bias=False)
        self.conv3d_1 = nn.Conv3d(20, 20, 3)
        self.conv3d_2 = nn.Conv3d(20, 30, 4)
        self.conv3d_3 = nn.Conv3d(30, 10, 4)
        self.pool3d_1 = nn.AvgPool3d(4, stride=4)

        projection_layers: list[nn.Module] = []
        current = representation_dim
        for hidden in representation_hidden_dims:
            projection_layers.extend((nn.Linear(current, hidden), nn.ELU()))
            current = hidden
        if current != projection_dim:
            projection_layers.extend((nn.Linear(current, projection_dim), nn.ELU()))
        if projection_name not in {"esm_mlp", "mlp"}:
            raise ValueError("projection_name must be 'esm_mlp' or 'mlp'")
        self.projection_name = projection_name
        self.add_module(projection_name, nn.Sequential(*projection_layers))

        self.conv1d_1 = nn.Conv1d(640 + one_body_dim + projection_dim, channels // 2, 1)
        self.conv2d_1 = nn.Conv2d(channels + two_body_dim, channels, 1)
        self.inorm_1 = nn.InstanceNorm2d(channels, eps=1e-6, affine=True)
        self.base_resnet = LegacyResNet(
            channels, chunks, "base_resnet", instance_norm=True, initial_projection=True
        )
        self.error_resnet = LegacyResNet(
            channels, 1, "error_resnet", initial_projection=True, extra_blocks=True
        )
        self.mask_resnet = LegacyResNet(
            channels, 1, "mask_resnet", initial_projection=True, extra_blocks=True
        )
        self.conv2d_error = nn.Conv2d(channels, 15, 1)
        self.conv2d_mask = nn.Conv2d(channels, 1, 1)

    def forward(
        self,
        voxel_indices: torch.Tensor,
        voxel_values: torch.Tensor,
        one_body: torch.Tensor,
        two_body: torch.Tensor,
        representation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if one_body.ndim != 2:
            raise ValueError(f"one_body must be [L, C], got {tuple(one_body.shape)}")
        length = one_body.shape[0]
        if tuple(representation.shape) != (length, self.representation_dim):
            raise ValueError(
                f"representation must be [{length}, {self.representation_dim}], got {tuple(representation.shape)}"
            )
        if two_body.ndim == 3:
            two_body = two_body.unsqueeze(0)
        if tuple(two_body.shape) != (1, self.two_body_dim, length, length):
            raise ValueError(
                f"two_body must be [1, {self.two_body_dim}, {length}, {length}], got {tuple(two_body.shape)}"
            )

        voxels = scatter_voxels(
            voxel_indices,
            voxel_values,
            (length, 24, 24, 24, self.residue_types),
        ).permute(0, 4, 1, 2, 3)
        output = F.elu(self.conv3d_1(self.retype(voxels)))
        output = F.elu(self.conv3d_2(output))
        output = F.elu(self.conv3d_3(output))
        output = self.pool3d_1(output)
        voxel_features = output.permute(0, 2, 3, 4, 1).flatten(start_dim=1)

        representation = self._modules[self.projection_name](representation)
        one_dimensional = torch.cat((voxel_features, one_body, representation), dim=1)
        one_dimensional = F.elu(self.conv1d_1(one_dimensional.T.unsqueeze(0)))

        rows = one_dimensional.unsqueeze(3).expand(-1, -1, -1, length)
        columns = one_dimensional.unsqueeze(2).expand(-1, -1, length, -1)
        pair = torch.cat((rows, columns, two_body), dim=1)
        pair = F.elu(self.inorm_1(self.conv2d_1(pair)))
        pair = F.elu(self.base_resnet(pair))

        error_features = F.elu(self.error_resnet(pair))
        estogram_logits = self.conv2d_error(error_features)
        estogram_logits = 0.5 * (estogram_logits + estogram_logits.transpose(2, 3))
        estogram = F.softmax(estogram_logits, dim=1)[0]

        mask_features = F.elu(self.mask_resnet(pair))
        mask_logits = self.conv2d_mask(mask_features)[:, 0]
        mask_logits = 0.5 * (mask_logits + mask_logits.transpose(1, 2))
        mask = torch.sigmoid(mask_logits)[0]
        local_lddt = calculate_lddt(estogram, mask)
        return {
            "estogram": estogram,
            "mask": mask,
            "local_lddt": local_lddt,
            "estogram_logits": estogram_logits,
            "mask_logits": mask_logits,
        }


def deepaccnet_loss(
    output: dict[str, torch.Tensor],
    estogram_target: torch.Tensor,
    mask_target: torch.Tensor,
    local_lddt_target: torch.Tensor,
    weights: tuple[float, float, float] = (1.0, 0.25, 10.0),
) -> tuple[torch.Tensor, dict[str, float]]:
    if estogram_target.ndim == 2:
        estogram_target = estogram_target.unsqueeze(0)
    if mask_target.ndim == 2:
        mask_target = mask_target.unsqueeze(0)
    estogram_loss = F.cross_entropy(output["estogram_logits"], estogram_target.long())
    mask_loss = F.binary_cross_entropy_with_logits(output["mask_logits"], mask_target.float())
    lddt_loss = F.mse_loss(output["local_lddt"], local_lddt_target.float())
    total = weights[0] * estogram_loss + weights[1] * mask_loss + weights[2] * lddt_loss
    metrics = {
        "loss": float(total.detach()),
        "estogram_loss": float(estogram_loss.detach()),
        "mask_loss": float(mask_loss.detach()),
        "lddt_loss": float(lddt_loss.detach()),
    }
    return total, metrics
