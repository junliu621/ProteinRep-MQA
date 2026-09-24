"""Residual blocks adapted from the MIT-licensed DeepAccNet implementation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResNet2D(nn.Module):
    def __init__(
        self,
        channels: int,
        chunks: int,
        *,
        instance_norm: bool = False,
        initial_projection: bool = False,
        extra_blocks: bool = False,
        dilation_cycle: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.initial = nn.Conv2d(channels, channels, 1) if initial_projection else nn.Identity()
        self.blocks = nn.ModuleList()
        for _ in range(chunks):
            for dilation in dilation_cycle:
                self.blocks.append(_ResidualBlock(channels, dilation, instance_norm))
        if extra_blocks:
            self.blocks.extend(_ResidualBlock(channels, 1, instance_norm) for _ in range(2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.initial(inputs)
        for block in self.blocks:
            output = block(output)
        return output


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, instance_norm: bool) -> None:
        super().__init__()
        middle = channels // 2
        norm = lambda size: nn.InstanceNorm2d(size, eps=1e-6, affine=True) if instance_norm else nn.Identity()
        self.norm1 = norm(channels)
        self.norm2 = norm(middle)
        self.norm3 = norm(middle)
        self.conv1 = nn.Conv2d(channels, middle, 1)
        self.conv2 = nn.Conv2d(middle, middle, 3, padding=dilation, dilation=dilation)
        self.conv3 = nn.Conv2d(middle, channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.conv1(F.elu(self.norm1(inputs)))
        output = self.conv2(F.elu(self.norm2(output)))
        output = self.conv3(F.elu(self.norm3(output)))
        return inputs + output


class LegacyResNet(nn.Module):
    """Checkpoint-compatible residual stack used by the original experiments."""

    def __init__(
        self,
        channels: int,
        chunks: int,
        name: str,
        *,
        instance_norm: bool = False,
        initial_projection: bool = False,
        extra_blocks: bool = False,
        dilation_cycle: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.channels = channels
        self.chunks = chunks
        self.name = name
        self.instance_norm = instance_norm
        self.initial_projection = initial_projection
        self.extra_blocks = extra_blocks
        self.dilation_cycle = dilation_cycle
        if initial_projection:
            self.add_module(f"resnet_{name}_init_proj", nn.Conv2d(channels, channels, 1))
        for chunk in range(chunks):
            for dilation in dilation_cycle:
                prefix = f"resnet_{name}_{chunk}_{dilation}"
                if instance_norm:
                    self.add_module(f"{prefix}_inorm_1", nn.InstanceNorm2d(channels, eps=1e-6, affine=True))
                    self.add_module(f"{prefix}_inorm_2", nn.InstanceNorm2d(channels // 2, eps=1e-6, affine=True))
                    self.add_module(f"{prefix}_inorm_3", nn.InstanceNorm2d(channels // 2, eps=1e-6, affine=True))
                self.add_module(f"{prefix}_conv2d_1", nn.Conv2d(channels, channels // 2, 1))
                self.add_module(
                    f"{prefix}_conv2d_2",
                    nn.Conv2d(channels // 2, channels // 2, 3, padding=dilation, dilation=dilation),
                )
                self.add_module(f"{prefix}_conv2d_3", nn.Conv2d(channels // 2, channels, 1))
        if extra_blocks:
            for index in range(2):
                prefix = f"resnet_{name}_extra{index}"
                if instance_norm:
                    self.add_module(f"{prefix}_inorm_1", nn.InstanceNorm2d(channels, eps=1e-6, affine=True))
                    self.add_module(f"{prefix}_inorm_2", nn.InstanceNorm2d(channels // 2, eps=1e-6, affine=True))
                    self.add_module(f"{prefix}_inorm_3", nn.InstanceNorm2d(channels // 2, eps=1e-6, affine=True))
                self.add_module(f"{prefix}_conv2d_1", nn.Conv2d(channels, channels // 2, 1))
                self.add_module(f"{prefix}_conv2d_2", nn.Conv2d(channels // 2, channels // 2, 3, padding=1))
                self.add_module(f"{prefix}_conv2d_3", nn.Conv2d(channels // 2, channels, 1))

    def _block(self, inputs: torch.Tensor, prefix: str) -> torch.Tensor:
        output = inputs
        if self.instance_norm:
            output = self._modules[f"{prefix}_inorm_1"](output)
        output = self._modules[f"{prefix}_conv2d_1"](F.elu(output))
        if self.instance_norm:
            output = self._modules[f"{prefix}_inorm_2"](output)
        output = self._modules[f"{prefix}_conv2d_2"](F.elu(output))
        if self.instance_norm:
            output = self._modules[f"{prefix}_inorm_3"](output)
        output = self._modules[f"{prefix}_conv2d_3"](F.elu(output))
        return inputs + output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        if self.initial_projection:
            output = self._modules[f"resnet_{self.name}_init_proj"](output)
        for chunk in range(self.chunks):
            for dilation in self.dilation_cycle:
                output = self._block(output, f"resnet_{self.name}_{chunk}_{dilation}")
        if self.extra_blocks:
            for index in range(2):
                output = self._block(output, f"resnet_{self.name}_extra{index}")
        return output
