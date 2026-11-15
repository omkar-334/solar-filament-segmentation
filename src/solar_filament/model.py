from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _groups(channels: int) -> int:
    for n in (8, 4, 2, 1):
        if channels % n == 0:
            return n
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    """Compact one-channel U-Net for high-resolution grayscale tiles."""

    def __init__(self, base_channels: int = 24):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, 2 * c)
        self.enc3 = ConvBlock(2 * c, 4 * c)
        self.enc4 = ConvBlock(4 * c, 8 * c)
        self.bottleneck = ConvBlock(8 * c, 16 * c)
        self.pool = nn.MaxPool2d(2)
        self.dec4 = ConvBlock(16 * c + 8 * c, 8 * c)
        self.dec3 = ConvBlock(8 * c + 4 * c, 4 * c)
        self.dec2 = ConvBlock(4 * c + 2 * c, 2 * c)
        self.dec1 = ConvBlock(2 * c + c, c)
        self.head = nn.Conv2d(c, 1, 1)

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((x, skip), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        x = self.dec4(self._up(b, e4))
        x = self.dec3(self._up(x, e3))
        x = self.dec2(self._up(x, e2))
        x = self.dec1(self._up(x, e1))
        return self.head(x)


class ResNet34SegmentationModel(nn.Module):
    """Grayscale model with an ImageNet-pretrained ResNet-34 encoder."""

    def __init__(
        self,
        pretrained_encoder: bool = False,
        decoder: str = "unet",
        output_channels: int = 1,
    ):
        super().__init__()
        self.output_channels = output_channels
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise RuntimeError(
                "ResNet34 segmentation architectures need segmentation-models-pytorch"
            ) from exc
        if decoder not in {"unet", "unetplusplus"}:
            raise ValueError(f"Unsupported ResNet34 decoder: {decoder}")
        decoder_class = smp.Unet if decoder == "unet" else smp.UnetPlusPlus
        self.network = decoder_class(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained_encoder else None,
            in_channels=3,
            classes=output_channels,
            decoder_channels=(256, 128, 64, 32, 16),
        )
        self.register_buffer(
            "imagenet_mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None]
        )
        self.register_buffer(
            "imagenet_std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = (x - self.imagenet_mean) / self.imagenet_std
        return self.network(x)


def build_model(
    architecture: str = "small_unet",
    base_channels: int = 24,
    pretrained_encoder: bool = False,
) -> nn.Module:
    if architecture == "small_unet":
        return SmallUNet(base_channels=base_channels)
    if architecture == "resnet34_unet":
        return ResNet34SegmentationModel(
            pretrained_encoder=pretrained_encoder, decoder="unet"
        )
    if architecture == "resnet34_unetpp":
        return ResNet34SegmentationModel(
            pretrained_encoder=pretrained_encoder, decoder="unetplusplus"
        )
    if architecture == "resnet34_unetpp_spine":
        # Channels: foreground logit, spine logit, then an 8-D pixel embedding.
        return ResNet34SegmentationModel(
            pretrained_encoder=pretrained_encoder,
            decoder="unetplusplus",
            output_channels=10,
        )
    if architecture == "ijepa_vith14_unet":
        from .ijepa_seg import IJEPAUNet

        return IJEPAUNet(pretrained=pretrained_encoder)
    raise ValueError(f"Unsupported architecture: {architecture}")


def binary_dice_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    denominator = probs.sum(dims) + targets.sum(dims)
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice_loss


def spine_heatmap_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Weighted BCE plus Dice for sparse rasterized centerline supervision."""
    pos_weight = torch.tensor(12.0, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    denominator = probs.sum(dims) + targets.sum(dims)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return 0.5 * bce + dice


def discriminative_instance_loss(
    embeddings: torch.Tensor,
    instance_ids: torch.Tensor,
    variance_margin: float = 0.5,
    distance_margin: float = 1.5,
    max_instances_per_crop: int = 64,
) -> torch.Tensor:
    """Pull same-instance pixels together and push distinct instance means apart.

    ``instance_ids`` uses zero for background and 65535 for ambiguous overlaps.
    The per-crop cap bounds work for unusually crowded patches.
    """
    batch_losses = []
    for batch_index in range(embeddings.shape[0]):
        labels = instance_ids[batch_index]
        valid = (labels > 0) & (labels < 65535)
        ids = torch.unique(labels[valid])
        if ids.numel() == 0:
            continue
        if ids.numel() > max_instances_per_crop:
            areas = torch.stack([(labels == object_id).sum() for object_id in ids])
            ids = ids[torch.topk(areas, max_instances_per_crop).indices]

        means = []
        variance_terms = []
        feature_map = embeddings[batch_index].permute(1, 2, 0)
        for object_id in ids:
            mask = labels == object_id
            vectors = feature_map[mask]
            mean = vectors.mean(dim=0)
            means.append(mean)
            distances = torch.linalg.vector_norm(vectors - mean[None, :], dim=1)
            variance_terms.append(F.relu(distances - variance_margin).square().mean())

        means_tensor = torch.stack(means)
        variance = torch.stack(variance_terms).mean()
        if means_tensor.shape[0] > 1:
            pair_distances = torch.pdist(means_tensor, p=2)
            distance = F.relu(distance_margin - pair_distances).square().mean()
        else:
            distance = embeddings[batch_index].sum() * 0.0
        regularization = torch.linalg.vector_norm(means_tensor, dim=1).mean()
        batch_losses.append(variance + distance + 1e-3 * regularization)
    if not batch_losses:
        return embeddings.sum() * 0.0
    return torch.stack(batch_losses).mean()
