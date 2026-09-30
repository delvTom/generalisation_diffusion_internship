"""
Standalone CelebA64 time-conditioned U-Net.

This file mirrors the repository architecture in
`config/archi_config/unet_celeba64.yaml` without depending on any local `src.*`
modules. The module names are kept compatible with the original implementation,
so checkpoints saved from the repo can be loaded with `model.load_state_dict`.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out


def _calculate_correct_fan(tensor: torch.Tensor, mode: str) -> int:
    mode = mode.lower()
    valid_modes = {"fan_in", "fan_out", "fan_avg"}
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def kaiming_uniform_(tensor: torch.Tensor, gain: float = 1.0, mode: str = "fan_in") -> torch.Tensor:
    fan = _calculate_correct_fan(tensor, mode)
    var = gain / max(1.0, fan)
    bound = math.sqrt(3.0 * var)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def linear(
    in_channels: int,
    out_channels: int,
    init_scale: float = 1.0,
    bias: bool = True,
    init_type: str = "kaiming_unif",
) -> nn.Linear:
    layer = nn.Linear(in_channels, out_channels, bias=bias)
    if init_type == "kaiming_unif":
        kaiming_uniform_(layer.weight, gain=1e-10 if init_scale == 0 else init_scale, mode="fan_avg")
    elif init_type == "xavier_unif":
        nn.init.xavier_uniform_(layer.weight)
    elif init_type == "normal":
        nn.init.normal_(layer.weight, std=0.02)
    elif init_type == "zero":
        nn.init.constant_(layer.weight, 0)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


def conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    bias: bool = True,
    init_type: str = "kaiming_unif",
    init_scale: float = 1.0,
) -> nn.Conv1d:
    layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
    if init_type == "zero":
        nn.init.zeros_(layer.weight)
    elif init_type == "kaiming_unif":
        kaiming_uniform_(layer.weight, gain=1e-10 if init_scale == 0 else init_scale, mode="fan_avg")
    elif init_type == "xavier_unif":
        nn.init.xavier_uniform_(layer.weight)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


def conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int] = (3, 3),
    stride: int = 1,
    dilation: int = 1,
    padding: int = 1,
    bias: bool = True,
    padding_mode: str = "zeros",
    init_type: str = "kaiming_unif",
    init_scale: float = 1.0,
) -> nn.Conv2d:
    layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        padding_mode=padding_mode,
    )
    if init_type == "kaiming_unif":
        kaiming_uniform_(layer.weight, gain=1e-10 if init_scale == 0 else init_scale, mode="fan_avg")
    elif init_type == "xavier_unif":
        weight = layer.weight.data
        nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


def downsample(in_ch: int, with_conv: bool) -> nn.Module:
    if with_conv:
        return conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=2)
    return nn.AvgPool2d(2, 2)


def upsample(in_ch: int, with_conv: bool) -> nn.Sequential:
    up = nn.Sequential()
    up.add_module("up_nn", nn.Upsample(scale_factor=2, mode="nearest"))
    if with_conv:
        up.add_module("up_conv", conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=1))
    return up


def label_embedder(num_classes: int, emb_dim: int, init_type: str = "default") -> nn.Embedding:
    embedder = nn.Embedding(num_classes, emb_dim)
    if init_type == "normal":
        nn.init.normal_(embedder.weight, std=0.2)
    return embedder


def sinusoidal_embedding(timesteps: torch.Tensor, embed_dim: int, max_period: int = 10000) -> torch.Tensor:
    half_dim = embed_dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        / half_dim
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embed_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        act: nn.Module = nn.SiLU(),
        init_type: str = "kaiming_unif",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.main = nn.Sequential(
            linear(embedding_dim, hidden_dim, init_type=init_type),
            act,
            linear(hidden_dim, output_dim, init_type=init_type),
        )

    def forward(self, temp: torch.Tensor) -> torch.Tensor:
        temb = sinusoidal_embedding(temp, self.embedding_dim)
        return self.main(temb)


class TimeBlock(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TimeSequential(nn.Sequential, TimeBlock):
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimeBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        normalization: type[nn.Module] | callable = nn.Identity,
        qk_normalization: type[nn.Module] | callable = nn.Identity,
        attn_use_res: bool = True,
        init_type_qkv: str = "kaiming_unif",
        init_type_proj: str = "zero",
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
            self.num_head_channels = channels // num_heads
        else:
            if channels % num_head_channels != 0:
                raise ValueError(f"channels {channels} is not divisible by num_head_channels {num_head_channels}")
            self.num_heads = channels // num_head_channels
            self.num_head_channels = num_head_channels

        self.normalization = normalization(channels)
        self.q_normalization = qk_normalization(self.num_head_channels)
        self.k_normalization = qk_normalization(self.num_head_channels)
        self.qkv = conv1d(channels, channels * 3, kernel_size=1, init_type=init_type_qkv)
        self.proj_out = conv1d(channels, channels, kernel_size=1, init_type=init_type_proj)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_res = attn_use_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_image = x.ndim == 4
        is_vit = x.ndim == 3 and x.shape[-1] == self.channels
        if is_image:
            bs, c, *spatial = x.shape
            x_in = x.reshape(bs, c, -1)
        elif is_vit:
            bs, _, _ = x.shape
            x_in = x.transpose(1, 2)
            spatial = []
        else:
            raise ValueError(f"Unsupported attention input shape: {tuple(x.shape)}")

        qkv = self.qkv(self.normalization(x_in))
        _, width, length = qkv.shape
        ch = width // (3 * self.num_heads)
        q, k, v = qkv.reshape(bs, self.num_heads, ch * 3, length).split(ch, dim=2)
        q = self.q_normalization(q.permute(0, 1, 3, 2))
        k = self.k_normalization(k.permute(0, 1, 3, 2))
        v = v.permute(0, 1, 3, 2)

        h = nn.functional.scaled_dot_product_attention(q, k, v)
        h = h.permute(0, 1, 3, 2).reshape(bs, -1, length)
        h = self.proj_out(h)
        out = x_in + h if self.use_res else h

        if is_image:
            return out.reshape(bs, c, *spatial)
        return out.transpose(1, 2)


class ResBlock(TimeBlock):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        out_channels: Optional[int] = None,
        use_residual: bool = True,
        use_conv_residual: bool = False,
        normalization: type[nn.Module] | callable = nn.Identity,
    ):
        super().__init__()
        self.channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.use_residual = use_residual
        self.in_layers = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            conv2d(in_channels, self.out_channels, kernel_size=3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(temb_channels, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout2d(p=dropout),
            conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, init_scale=0.0),
        )

        if use_residual:
            if use_conv_residual:
                self.skip_connection = conv2d(in_channels, self.out_channels, kernel_size=1, padding=0)
            elif in_channels != self.out_channels:
                self.skip_connection = conv2d(in_channels, self.out_channels, kernel_size=1, padding=0)
            else:
                self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)[:, :, None, None]
        h = h + emb_out
        h = self.out_layers(h)
        if self.use_residual:
            h = self.skip_connection(x) + h
        return h


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        num_res_blocks: int,
        attention_levels: Iterable[int],
        channel_mult: Iterable[int],
        dropout: float = 0.0,
        num_classes: Optional[int] = None,
        use_residual: bool = True,
        attn_use_residual: bool = True,
        use_conv_residual: bool = False,
        use_conv_rescale: bool = True,
        attention_num_heads: int = 1,
        attention_num_head_channels: int = 1,
        use_attention_middle: bool = True,
        use_middle_conv_residual: bool = True,
        normalization: Optional[str] = None,
    ):
        super().__init__()
        if normalization == "group-norm":
            normalization_layer = lambda channels: nn.GroupNorm(32, channels)
        else:
            normalization_layer = nn.Identity

        attention_levels = set(attention_levels)
        channel_mult = list(channel_mult)
        self.num_classes = num_classes
        self.num_resolutions = len(channel_mult)

        time_embed_dim = base_channels * 4
        self.time_embed = TimestepEmbedding(base_channels, time_embed_dim, time_embed_dim)
        if self.num_classes is not None:
            self.label_embed = label_embedder(num_classes, time_embed_dim)

        ch = base_channels
        self.in_layer = conv2d(in_channels, ch)
        ds = 1
        unet_chs = [ch]

        self.down_modules = []
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * base_channels),
                        use_residual=use_residual,
                        use_conv_residual=use_conv_residual,
                        normalization=normalization_layer,
                    )
                ]
                ch = int(mult * base_channels)
                if ds in attention_levels:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=attention_num_heads,
                            num_head_channels=attention_num_head_channels,
                            attn_use_res=attn_use_residual,
                            normalization=normalization_layer,
                        )
                    )
                self.down_modules.append(TimeSequential(*layers))
                unet_chs.append(ch)

            if level != len(channel_mult) - 1:
                self.down_modules.append(TimeSequential(downsample(ch, with_conv=use_conv_rescale)))
                unet_chs.append(ch)
                ds *= 2

        self.down_modules = nn.ModuleList(self.down_modules)

        self.middle_module = TimeSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_residual=use_residual,
                use_conv_residual=use_middle_conv_residual,
                normalization=normalization_layer,
            ),
            AttentionBlock(
                ch,
                num_heads=attention_num_heads,
                num_head_channels=attention_num_head_channels,
                attn_use_res=attn_use_residual,
                normalization=normalization_layer,
            )
            if use_attention_middle
            else nn.Identity(),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_residual=use_residual,
                use_conv_residual=use_middle_conv_residual,
                normalization=normalization_layer,
            ),
        )

        self.up_modules = []
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                in_ch = ch + unet_chs.pop()
                layers = [
                    ResBlock(
                        in_ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(base_channels * mult),
                        use_residual=use_residual,
                        use_conv_residual=use_conv_residual,
                        normalization=normalization_layer,
                    )
                ]
                ch = int(mult * base_channels)
                if ds in attention_levels:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=attention_num_heads,
                            num_head_channels=attention_num_head_channels,
                            attn_use_res=attn_use_residual,
                            normalization=normalization_layer,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(upsample(ch, with_conv=use_conv_rescale))
                    ds //= 2
                self.up_modules.append(TimeSequential(*layers))

        self.up_modules = nn.ModuleList(self.up_modules)
        self.out_layer = nn.Sequential(
            normalization_layer(ch),
            nn.SiLU(),
            conv2d(ch, out_channels, init_scale=0.0),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        timesteps = t
        if (y is not None) != (self.num_classes is not None):
            raise ValueError("must specify y if and only if the model is class-conditional")

        while timesteps.dim() > 1:
            timesteps = timesteps[:, 0]
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(x.shape[0])

        emb = self.time_embed(timesteps)
        if self.num_classes is not None:
            if y.shape != (x.shape[0],):
                raise ValueError(f"expected y shape {(x.shape[0],)}, got {tuple(y.shape)}")
            emb = emb + self.label_embed(y)

        h = self.in_layer(x)
        hs = [h]
        for module in self.down_modules:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_module(h, emb)

        for module in self.up_modules:
            h_prev = hs.pop()
            h = torch.cat([h, h_prev], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        return self.out_layer(h)


def create_unet_celeba64(use_residual: bool = True) -> UNetModel:
    """Create the `unet_celeba64` architecture from the repo config."""
    return UNetModel(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        num_res_blocks=6,
        attention_levels=[3, 4],
        channel_mult=[1, 2, 4, 8],
        dropout=0.0,
        num_classes=None,
        use_residual=use_residual,
        use_conv_residual=False,
        use_conv_rescale=True,
        attention_num_heads=4,
        attention_num_head_channels=64,
        use_attention_middle=True,
        use_middle_conv_residual=True,
        normalization="group-norm",
    )
