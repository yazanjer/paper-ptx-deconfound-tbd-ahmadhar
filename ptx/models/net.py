"""Segmentation-classification network with the deconfounding components.

Components (all switchable from the config):
    C2  adv_device   gradient-reversal head predicting 'treated' (tube/device) from bottleneck features
    C3  aux_tube_seg extra segmentation channel for the chest tube
    C4  mask_pool    image score = top-k mean of pneumothorax segmentation logits (else global head)
    --  aux_tube_cls image-level tube classifier WITHOUT reversal (Bandaru et al. 2026 baseline)
C1 (inpainting) lives in the dataset; C5 (size) is post-hoc from the predicted mask (ptx/eval/size.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam=1.0):
    return _GRL.apply(x, lam)


class PTXNet(nn.Module):
    def __init__(self, arch="unet", encoder="resnet34", pretrained="imagenet", aux_tube_seg=False,
                 mask_pool=True, adv_device=False, adv_lambda=0.1, aux_tube_cls=False, topk_frac=0.005):
        super().__init__()
        import segmentation_models_pytorch as smp
        classes = 2 if aux_tube_seg else 1
        ctor = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus, "fpn": smp.FPN,
                "segformer": smp.Segformer, "deeplabv3plus": smp.DeepLabV3Plus}[arch]
        self.seg = ctor(encoder_name=encoder, encoder_weights=pretrained, classes=classes, in_channels=3)
        c = self.seg.encoder.out_channels[-1]
        self.aux_tube_seg, self.mask_pool, self.adv_device, self.aux_tube_cls = aux_tube_seg, mask_pool, adv_device, aux_tube_cls
        self.adv_lambda, self.topk_frac = adv_lambda, topk_frac
        self.global_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(c, 1))
        self.pool_affine = nn.Parameter(torch.tensor([1.0, 0.0]))
        if adv_device:
            self.adv_head = nn.Sequential(nn.Linear(c, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        if aux_tube_cls:
            self.aux_cls_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(c, 1))

    def forward(self, x):
        feats = self.seg.encoder(x)
        dec = self.seg.decoder(feats)
        seg = self.seg.segmentation_head(dec)
        if seg.shape[-1] != x.shape[-1]:
            seg = F.interpolate(seg, size=x.shape[-2:], mode="bilinear", align_corners=False)
        pooled = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)
        out = {"seg_ptx": seg[:, :1], "feat": pooled}
        if self.aux_tube_seg:
            out["seg_tube"] = seg[:, 1:2]
        if self.mask_pool:
            flat = seg[:, 0].flatten(1)
            k = max(1, int(self.topk_frac * flat.shape[1]))
            s = flat.topk(k, dim=1).values.mean(1)
            out["cls"] = self.pool_affine[0] * s + self.pool_affine[1]
        else:
            out["cls"] = self.global_head(pooled).squeeze(1)
        if self.adv_device:
            out["adv"] = self.adv_head(grad_reverse(pooled, self.adv_lambda)).squeeze(1)
        if self.aux_tube_cls:
            out["aux_cls"] = self.aux_cls_head(pooled).squeeze(1)
        return out


class FrozenProbe(nn.Module):
    """Foundation-model baseline: frozen encoder + linear classification head (no segmentation).

    backbone: 'rad-dino' (microsoft/rad-dino via transformers) or any timm model name.
    """

    def __init__(self, backbone="rad-dino", finetune=False):
        super().__init__()
        self.kind = backbone
        if backbone == "rad-dino":
            from transformers import AutoModel
            self.enc = AutoModel.from_pretrained("microsoft/rad-dino")
            dim = self.enc.config.hidden_size
        else:
            import timm
            self.enc = timm.create_model(backbone, pretrained=True, num_classes=0)
            dim = self.enc.num_features
        for p in self.enc.parameters():
            p.requires_grad = finetune
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        if self.kind == "rad-dino":
            x = F.interpolate(x, size=(518, 518), mode="bilinear", align_corners=False)
            f = self.enc(pixel_values=x).pooler_output
        else:
            f = self.enc(x)
        return {"cls": self.head(f).squeeze(1), "feat": f}


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    if m.get("type", "ptxnet") == "probe":
        return FrozenProbe(m.get("backbone", "rad-dino"), m.get("finetune", False))
    return PTXNet(arch=m.get("arch", "unet"), encoder=m.get("encoder", "resnet34"),
                  pretrained=m.get("pretrained", "imagenet"), aux_tube_seg=m.get("aux_tube_seg", False),
                  mask_pool=m.get("mask_pool", True), adv_device=m.get("adv_device", False),
                  adv_lambda=m.get("adv_lambda", 0.1), aux_tube_cls=m.get("aux_tube_cls", False),
                  topk_frac=m.get("topk_frac", 0.005))
