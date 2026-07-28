"""MHRGait++: end-to-end DeepGaitV2 and MHRGait joint training."""

import torch
import torch.nn as nn
from einops import rearrange

from ..base_model import BaseModel
from ..modules import (
    BasicBlock2D,
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateBNNecks,
    SeparateFCs,
    SetBlockWrapper,
    conv1x1,
    conv3x3,
)
from .deepgaitv2 import blocks_map
from .mhrgait import MHRGait


def build_mhr_backbone(model_cfg):
    """Build MHRGait without creating a second OpenGait runtime."""
    model = MHRGait.__new__(MHRGait)
    nn.Module.__init__(model)
    model.build_network(model_cfg)
    return model


class MHRGaitPlusPlus(BaseModel):
    """Jointly optimize 16 silhouette parts and two semantic MHR parts."""

    def build_network(self, model_cfg):
        self._build_silhouette_encoder(model_cfg)
        self.sil_FCs = self.FCs
        del self.FCs
        del self.BNNecks

        self.mhr = build_mhr_backbone(model_cfg["mhr_model_cfg"])
        if self.mhr.FCs.p != 2:
            raise ValueError("MHRGait must provide body and hand parts")
        if self.mhr.FCs.fc_bin.size(-1) != self.sil_FCs.fc_bin.size(-1):
            raise ValueError(
                "Silhouette and MHR parts must share one embedding dimension"
            )

        del self.mhr.BNNecks
        self.BNNecks = SeparateBNNecks(
            parts_num=18,
            in_channels=self.mhr.FCs.fc_bin.size(-1),
            class_num=model_cfg["SeparateBNNecks"]["class_num"],
        )

    def _build_silhouette_encoder(self, model_cfg):
        backbone_cfg = model_cfg["SilhouetteBackbone"]
        mode = backbone_cfg["mode"]
        if mode not in blocks_map:
            raise ValueError(f"Unsupported silhouette mode: {mode}")
        block = blocks_map[mode]
        layers = backbone_cfg["layers"]
        channels = backbone_cfg["channels"]
        if len(layers) != 4 or len(channels) != 4:
            raise ValueError("SilhouetteBackbone requires four stages")
        strides = (
            ([1, 1], [1, 2, 2], [1, 2, 2], [1, 1, 1])
            if mode == "3d"
            else ([1, 1], [2, 2], [2, 2], [1, 1])
        )

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(backbone_cfg["in_channels"], self.inplanes, 1),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
            )
        )
        self.layer1 = SetBlockWrapper(
            self._make_sil_layer(
                BasicBlock2D, channels[0], strides[0], layers[0], mode
            )
        )
        self.layer2 = self._make_sil_layer(
            block, channels[1], strides[1], layers[1], mode
        )
        self.layer3 = self._make_sil_layer(
            block, channels[2], strides[2], layers[2], mode
        )
        self.layer4 = self._make_sil_layer(
            block, channels[3], strides[3], layers[3], mode
        )
        if mode == "2d":
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(
            16,
            channels[2],
            class_num=model_cfg["SeparateBNNecks"]["class_num"],
        )
        self.sil_TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def _make_sil_layer(self, block, planes, stride, blocks_num, mode):
        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == "3d":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            elif mode == "2d":
                downsample = nn.Sequential(
                    conv1x1(
                        self.inplanes,
                        planes * block.expansion,
                        stride=stride,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            elif mode == "p3d":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=[1, *stride],
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            else:
                raise ValueError(f"Unsupported silhouette mode: {mode}")
        else:
            downsample = lambda x: x

        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.inplanes = planes * block.expansion
        unit_stride = [1, 1] if mode in ("2d", "p3d") else [1, 1, 1]
        for _ in range(1, blocks_num):
            layers.append(
                block(self.inplanes, planes, stride=unit_stride)
            )
        return nn.Sequential(*layers)

    def _encode_silhouette(self, sils, seqL):
        if sils.ndim == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = sils.transpose(1, 2).contiguous()
        if sils.size(-1) not in (44, 88):
            raise ValueError(
                f"Expected cropped silhouette width 44 or 88, got {sils.size(-1)}"
            )
        out = self.layer4(
            self.layer3(
                self.layer2(self.layer1(self.layer0(sils)))
            )
        )
        pooled = self.sil_TP(out, seqL, options={"dim": 2})[0]
        return self.HPP(pooled), sils

    def _encode_mhr(self, mhr, seqL):
        return self.mhr.FCs(self.mhr.encode_parts(mhr, seqL))

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        if len(ipts) != 2:
            raise ValueError(
                f"Expected [silhouette, MHR] streams, got {len(ipts)}"
            )
        silhouette_raw, sils = self._encode_silhouette(ipts[0], seqL)
        silhouette_parts = self.sil_FCs(silhouette_raw)
        mhr_parts = self._encode_mhr(ipts[1], seqL)
        embeddings = torch.cat((silhouette_parts, mhr_parts), dim=-1)
        _, logits = self.BNNecks(embeddings)
        return {
            "training_feat": {
                "triplet": {"embeddings": embeddings, "labels": labs},
                "softmax": {"logits": logits, "labels": labs},
            },
            "visual_summary": {
                "image/sils": rearrange(
                    sils, "n c s h w -> (n s) c h w"
                )
            },
            "inference_feat": {"embeddings": embeddings},
        }
