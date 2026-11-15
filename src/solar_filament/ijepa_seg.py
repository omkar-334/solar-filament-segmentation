from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


IJEPA_MODEL_ID = "facebook/ijepa_vith14_1k"


def _norm_groups(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.GELU(),
        )


class IJEPAUNet(nn.Module):
    """Frozen/partially tuned I-JEPA ViT encoder with a detail-preserving decoder."""

    output_channels = 1

    def __init__(self, pretrained: bool = True, trainable_encoder_blocks: int = 4):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise RuntimeError("I-JEPA segmentation needs transformers") from exc

        if pretrained:
            self.encoder = AutoModel.from_pretrained(
                IJEPA_MODEL_ID, attn_implementation="sdpa"
            )
        else:
            config = AutoConfig.from_pretrained(IJEPA_MODEL_ID)
            self.encoder = AutoModel.from_config(
                config, attn_implementation="sdpa"
            )

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        layers = self.encoder.encoder.layer
        for layer in layers[-trainable_encoder_blocks:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        for parameter in self.encoder.layernorm.parameters():
            parameter.requires_grad_(True)

        width = self.encoder.config.hidden_size
        self.feature_projection = nn.Sequential(
            nn.Conv2d(width, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.GELU(),
        )

        self.local_full = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.local_half = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1, bias=False),
            nn.GroupNorm(8, 24),
            nn.GELU(),
        )
        self.local_quarter = _ConvBlock(1, 32)
        self.local_eighth = _ConvBlock(1, 48)

        self.decode_eighth = _ConvBlock(256 + 48, 160)
        self.decode_quarter = _ConvBlock(160 + 32, 96)
        self.decode_half = _ConvBlock(96 + 24, 48)
        self.decode_full = _ConvBlock(48 + 16, 32)
        self.mask_head = nn.Conv2d(32, 1, 1)

        self.register_buffer(
            "imagenet_mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None]
        )
        self.register_buffer(
            "imagenet_std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None]
        )

    def encoder_parameters(self):
        return (parameter for parameter in self.encoder.parameters() if parameter.requires_grad)

    def decoder_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("encoder."):
                yield parameter

    @staticmethod
    def _resize(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

    def forward(self, grayscale: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = grayscale.shape
        rgb = grayscale.expand(-1, 3, -1, -1)
        rgb = (rgb - self.imagenet_mean) / self.imagenet_std
        tokens = self.encoder(
            pixel_values=rgb,
            interpolate_pos_encoding=True,
        ).last_hidden_state
        grid_h = height // self.encoder.config.patch_size
        grid_w = width // self.encoder.config.patch_size
        features = tokens.transpose(1, 2).reshape(batch, -1, grid_h, grid_w)
        features = self.feature_projection(features)

        half_size = (max(1, height // 2), max(1, width // 2))
        quarter_size = (max(1, height // 4), max(1, width // 4))
        eighth_size = (max(1, height // 8), max(1, width // 8))
        local_half = self.local_half(self._resize(grayscale, half_size))
        local_quarter = self.local_quarter(self._resize(grayscale, quarter_size))
        local_eighth = self.local_eighth(self._resize(grayscale, eighth_size))

        value = self._resize(features, eighth_size)
        value = self.decode_eighth(torch.cat((value, local_eighth), dim=1))
        value = self._resize(value, quarter_size)
        value = self.decode_quarter(torch.cat((value, local_quarter), dim=1))
        value = self._resize(value, half_size)
        value = self.decode_half(torch.cat((value, local_half), dim=1))
        value = self._resize(value, (height, width))
        value = self.decode_full(
            torch.cat((value, self.local_full(grayscale)), dim=1)
        )
        return self.mask_head(value)
