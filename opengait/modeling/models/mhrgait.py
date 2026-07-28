"""MHRGait: semantic MHR-Mixer with adaptive pooling and temporal modeling."""

import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import PackSequenceWrapper, SeparateBNNecks, SeparateFCs


MHR_DIM = 389
BODY76_PARTS = (
    ("torso_head", tuple(range(3, 27))),
    ("right_arm", tuple(range(27, 37))),
    ("left_arm", tuple(range(37, 47))),
    ("right_leg_foot", tuple(range(47, 56)) + tuple(range(123, 127))),
    ("left_leg_foot", tuple(range(56, 65)) + tuple(range(119, 123))),
    ("body_flexibility", tuple(range(127, 133))),
)
HAND_PARTS = (
    ("left_hand", tuple(range(136, 190))),
    ("right_hand", tuple(range(190, 244))),
)


class SemanticMixerBlock(nn.Module):
    """Mix semantic tokens and feature channels with residual MLPs."""

    def __init__(
        self,
        hidden_dim,
        num_tokens,
        token_mlp_ratio,
        dropout,
        channel_mlp_ratio=4,
    ):
        super().__init__()
        token_hidden_dim = max(1, int(num_tokens * token_mlp_ratio))
        channel_hidden_dim = max(1, int(hidden_dim * channel_mlp_ratio))
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_expand = nn.Linear(num_tokens, token_hidden_dim)
        self.token_project = nn.Linear(token_hidden_dim, num_tokens)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_expand = nn.Linear(hidden_dim, channel_hidden_dim)
        self.channel_project = nn.Linear(channel_hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        tokens = self.token_norm(x).transpose(2, 3)
        tokens = self.token_project(
            self.dropout(self.activation(self.token_expand(tokens)))
        )
        x = x + self.dropout(tokens.transpose(2, 3))
        channels = self.channel_project(
            self.dropout(self.activation(self.channel_expand(self.channel_norm(x))))
        )
        return x + self.dropout(channels)


class SemanticBranchMixerEncoder(nn.Module):
    """Encode one MHR field as semantically grouped parameter tokens."""

    def __init__(
        self,
        parts,
        hidden_dim,
        embed_dim,
        mixer_layers,
        token_mlp_ratio,
        channel_mlp_ratio,
        dropout,
    ):
        super().__init__()
        self.node_names = tuple(name for name, _ in parts)
        self.node_indices = tuple(indices for _, indices in parts)
        self.param_nodes = len(self.node_indices)
        self.hub_nodes = 1
        self.num_nodes = self.param_nodes + self.hub_nodes

        self.node_projs = nn.ModuleList(
            [nn.Linear(len(indices), hidden_dim) for indices in self.node_indices]
        )
        self.field_embed = nn.Embedding(1, hidden_dim)
        self.node_embed = nn.Embedding(self.num_nodes, hidden_dim)
        self.hub_embed = nn.Parameter(
            torch.zeros(1, 1, self.hub_nodes, hidden_dim)
        )
        self.mixer_blocks = nn.ModuleList(
            [
                SemanticMixerBlock(
                    hidden_dim,
                    self.num_nodes,
                    token_mlp_ratio,
                    dropout,
                    channel_mlp_ratio,
                )
                for _ in range(mixer_layers)
            ]
        )
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def encode_tokens(self, mhr):
        batch_size, frame_count, _ = mhr.shape
        nodes = torch.cat(
            [
                projection(
                    mhr.index_select(
                        -1, torch.as_tensor(indices, device=mhr.device)
                    )
                ).unsqueeze(2)
                for projection, indices in zip(
                    self.node_projs, self.node_indices
                )
            ],
            dim=2,
        )
        x = torch.cat(
            [
                nodes,
                self.hub_embed.expand(batch_size, frame_count, -1, -1),
            ],
            dim=2,
        )
        token_ids = torch.arange(self.num_nodes, device=mhr.device)
        field_ids = torch.zeros(
            self.num_nodes, device=mhr.device, dtype=torch.long
        )
        x = x + self.field_embed(field_ids).view(
            1, 1, self.num_nodes, -1
        )
        x = x + self.node_embed(token_ids).view(
            1, 1, self.num_nodes, -1
        )
        for block in self.mixer_blocks:
            x = block(x)
        return x

    def aggregate_tokens(self, tokens):
        physical_tokens = tokens[:, :, : self.param_nodes]
        weights = torch.softmax(
            self.pool_score(self.pool_norm(physical_tokens)), dim=2
        )
        pooled_tokens = (physical_tokens * weights).sum(dim=2)
        hub_token = tokens[:, :, self.param_nodes :].mean(dim=2)
        return self.frame_proj(
            torch.cat((pooled_tokens, hub_token), dim=-1)
        )

    def forward(self, mhr):
        return self.aggregate_tokens(self.encode_tokens(mhr))


class AdaptiveSemanticBranchMixerEncoder(SemanticBranchMixerEncoder):
    """Add learned attention pooling over physical semantic tokens."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.node_projs[0].out_features
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.pool_score = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.pool_score.weight)
        nn.init.zeros_(self.pool_score.bias)


def _sequence_lengths(seqL):
    if seqL is None:
        return None
    return [
        int(value)
        for value in torch.as_tensor(seqL).detach().cpu().reshape(-1).tolist()
    ]


class PackedTemporalBottleneck(nn.Module):
    """Residual TCN that never convolves across packed sequence boundaries."""

    def __init__(self, channels, reduction=4, kernel_size=9, dropout=0.1):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("temporal kernel_size must be odd")
        inner_channels = max(1, channels // reduction)
        self.down = nn.Linear(channels, inner_channels)
        self.down_norm = nn.LayerNorm(inner_channels)
        self.temporal = nn.Conv1d(
            inner_channels,
            inner_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.temporal_norm = nn.LayerNorm(inner_channels)
        self.up = nn.Linear(inner_channels, channels)
        self.up_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
        nn.init.zeros_(self.up_norm.weight)
        nn.init.zeros_(self.up_norm.bias)

    def _conv_segment(self, features):
        return self.temporal(features.transpose(1, 2)).transpose(1, 2)

    def _apply_temporal(self, features, seqL):
        lengths = _sequence_lengths(seqL)
        if lengths is None:
            return self._conv_segment(features)
        outputs = []
        start = 0
        for length in lengths:
            end = start + length
            outputs.append(self._conv_segment(features[:, start:end]))
            start = end
        if start != features.size(1):
            raise ValueError("Packed temporal lengths must cover all frames")
        return torch.cat(outputs, dim=1)

    def forward(self, features, seqL):
        residual = features
        branch = self.activation(self.down_norm(self.down(features)))
        branch = self.dropout(branch)
        branch = self._apply_temporal(branch, seqL)
        branch = self.activation(self.temporal_norm(branch))
        branch = self.up_norm(self.up(branch))
        return residual + branch


class MHRGait(BaseModel):
    """Two-part gait model for body76 and hand108 MHR parameters."""

    def build_network(self, model_cfg):
        hidden_dim = model_cfg.get("hidden_dim", 256)
        self.embed_dim = model_cfg.get("embed_dim", 256)
        encoder_args = (
            hidden_dim,
            self.embed_dim,
            model_cfg.get("mixer_layers", 4),
            model_cfg.get("token_mlp_ratio", 4),
            model_cfg.get("channel_mlp_ratio", 4),
            model_cfg.get("dropout", 0.2),
        )
        self.body_mixer = AdaptiveSemanticBranchMixerEncoder(
            BODY76_PARTS, *encoder_args
        )
        self.hand_mixer = AdaptiveSemanticBranchMixerEncoder(
            HAND_PARTS, *encoder_args
        )
        self.body_temporal = self._build_temporal_adapter(model_cfg)
        self.hand_temporal = self._build_temporal_adapter(model_cfg)
        self._build_pool(model_cfg)
        self._build_heads(model_cfg)

    def _build_temporal_adapter(self, model_cfg):
        return PackedTemporalBottleneck(
            channels=self.embed_dim,
            reduction=model_cfg.get("temporal_reduction", 4),
            kernel_size=model_cfg.get("temporal_kernel_size", 9),
            dropout=model_cfg.get("temporal_dropout", 0.1),
        )

    def _build_pool(self, model_cfg):
        pool = model_cfg.get("temporal_pool", "max")
        if pool != "max":
            raise ValueError("MHRGait release supports temporal_pool: max")
        self.TP = PackSequenceWrapper(torch.max)

    def _build_heads(self, model_cfg):
        fc_cfg = model_cfg["SeparateFCs"]
        neck_cfg = model_cfg["SeparateBNNecks"]
        if fc_cfg["in_channels"] != self.embed_dim or fc_cfg["parts_num"] != 2:
            raise ValueError(
                "SeparateFCs must match embed_dim and two MHR parts"
            )
        if (
            neck_cfg["in_channels"] != fc_cfg["out_channels"]
            or neck_cfg["parts_num"] != 2
        ):
            raise ValueError(
                "SeparateBNNecks must match SeparateFCs and two MHR parts"
            )
        self.FCs = SeparateFCs(**fc_cfg)
        self.BNNecks = SeparateBNNecks(**neck_cfg)

    def _pool(self, frames, seqL):
        return self.TP(
            frames.transpose(1, 2).contiguous(),
            seqL,
            options={"dim": 2},
        )[0]

    def encode_parts(self, mhr, seqL):
        body_frames = self.body_temporal(self.body_mixer(mhr), seqL)
        hand_frames = self.hand_temporal(self.hand_mixer(mhr), seqL)
        body = self._pool(body_frames, seqL)
        hand = self._pool(hand_frames, seqL)
        return torch.stack((body, hand), dim=-1)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        if len(ipts) != 1:
            raise ValueError(f"Expected one MHR stream, got {len(ipts)}")
        mhr = ipts[0]
        if mhr.ndim != 3 or mhr.size(-1) != MHR_DIM:
            raise ValueError(
                f"Expected MHR input [N, T, {MHR_DIM}], got {tuple(mhr.shape)}"
            )
        embed_1 = self.FCs(self.encode_parts(mhr, seqL))
        _, logits = self.BNNecks(embed_1)
        return {
            "training_feat": {
                "triplet": {"embeddings": embed_1, "labels": labs},
                "softmax": {"logits": logits, "labels": labs},
            },
            "visual_summary": {},
            "inference_feat": {"embeddings": embed_1},
        }
