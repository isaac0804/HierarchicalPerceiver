import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from utils import imshow

# Implementation of HiP-16 and HiP-256


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, expanding_factor=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, embed_dim * expanding_factor)
        self.linear2 = nn.Linear(embed_dim * expanding_factor, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(self.mha(x, x, x)[0] + x)
        x = self.norm2(F.gelu(self.linear2(F.gelu(self.linear1(x)))) + x)
        return x


class HiPLayer(nn.Module):
    def __init__(self, num_groups, num_attn_layers, num_latents, num_channels,
                 num_heads, input_channels=None):
        super().__init__()
        self.num_groups = num_groups
        self.num_attn_layers = num_attn_layers
        self.num_latents = num_latents
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.dot_scale = 1 / math.sqrt(num_channels)

        input_channels = num_channels if input_channels == None else input_channels
        self.linear = nn.Linear(input_channels, num_channels)
        self.learned_embedding = nn.Parameter(
            torch.randn(num_groups, num_latents, num_channels))
        self.attention = nn.MultiheadAttention(input_channels, num_heads)

        self.stem = nn.ModuleList(
            [SelfAttentionLayer(num_channels, num_heads)
             for _ in range(num_attn_layers)]
        )

    def forward(self, x):
        """
        Input
        -----
        x : torch.tensor (batch_size x num_tokens x num_channels)
        """
        B, N, C = x.shape
        x = x.view(-1, self.num_groups, N//self.num_groups, C)
        x = self.linear(x)
        # print(x.shape)

        attn = torch.einsum(
            "gkd,bghd->bgkh", self.learned_embedding, x) * self.dot_scale
        attn = torch.softmax(attn, dim=-2)
        x = torch.einsum("bgkh,bghd->bgkd", attn, x)
        # print(x.shape)

        x = x.view(-1, self.num_latents, self.num_channels)
        for _, layer in enumerate(self.stem):
            x = layer(x)
        # print(x.shape)
        x = x.view(B, self.num_groups * self.num_latents, self.num_channels)
        # print(x.shape)

        return x, attn


class HiP16_small(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                HiPLayer(num_groups=16, num_attn_layers=2, num_latents=32,
                         num_channels=64, num_heads=4, input_channels=32),
                HiPLayer(num_groups=4, num_attn_layers=2, num_latents=64,
                         num_channels=128, num_heads=8, input_channels=64),
                HiPLayer(num_groups=1, num_attn_layers=18, num_latents=128,
                         num_channels=256, num_heads=16, input_channels=128),
                HiPLayer(num_groups=1, num_attn_layers=2, num_latents=64,
                         num_channels=512, num_heads=32, input_channels=256),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                HiPLayer(num_groups=1, num_attn_layers=1, num_latents=128,
                         num_channels=256, num_heads=16, input_channels=512),
                HiPLayer(num_groups=4, num_attn_layers=1, num_latents=64,
                         num_channels=128, num_heads=8, input_channels=256),
                HiPLayer(num_groups=16, num_attn_layers=1, num_latents=32,
                         num_channels=64, num_heads=4, input_channels=128)
            ]
        )

    def forward(self, x):
        skips = []
        for _, layer in enumerate(self.encoder):
            skips.append(x)
            x = layer(x)[0]
        for _, layer in enumerate(self.decoder):
            x = layer(x)[0] + skips.pop()
        return x


class HiP16(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                HiPLayer(num_groups=16, num_attn_layers=2, num_latents=128,
                         num_channels=128, num_heads=4, input_channels=32),
                HiPLayer(num_groups=4, num_attn_layers=2, num_latents=256,
                         num_channels=256, num_heads=8, input_channels=128),
                HiPLayer(num_groups=1, num_attn_layers=18, num_latents=256,
                         num_channels=512, num_heads=16, input_channels=256),
                HiPLayer(num_groups=1, num_attn_layers=2, num_latents=64,
                         num_channels=1024, num_heads=32, input_channels=512),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                HiPLayer(num_groups=1, num_attn_layers=1, num_latents=256,
                         num_channels=512, num_heads=16, input_channels=1024),
                HiPLayer(num_groups=4, num_attn_layers=1, num_latents=256,
                         num_channels=256, num_heads=8, input_channels=512),
                HiPLayer(num_groups=16, num_attn_layers=1, num_latents=128,
                         num_channels=128, num_heads=4, input_channels=256)
            ]
        )

    def forward(self, x):
        skips = []
        for _, layer in enumerate(self.encoder):
            skips.append(x)
            x = layer(x)[0]
        for _, layer in enumerate(self.decoder):
            x = layer(x)[0] + skips.pop()
        return x


class HiP256(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                HiPLayer(num_groups=256, num_attn_layers=1, num_latents=32,
                         num_channels=64, num_heads=1, input_channels=16),
                HiPLayer(num_groups=64, num_attn_layers=1, num_latents=64,
                         num_channels=96, num_heads=2, input_channels=64),
                HiPLayer(num_groups=16, num_attn_layers=2, num_latents=128,
                         num_channels=128, num_heads=4, input_channels=96),
                HiPLayer(num_groups=4, num_attn_layers=2, num_latents=256,
                         num_channels=256, num_heads=8, input_channels=128),
                HiPLayer(num_groups=1, num_attn_layers=18, num_latents=256,
                         num_channels=512, num_heads=16, input_channels=256),
                HiPLayer(num_groups=1, num_attn_layers=2, num_latents=64,
                         num_channels=1024, num_heads=32, input_channels=512),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                HiPLayer(num_groups=1, num_attn_layers=1, num_latents=256,
                         num_channels=256, num_heads=16, input_channels=1024),
                HiPLayer(num_groups=4, num_attn_layers=1, num_latents=256,
                         num_channels=128, num_heads=8, input_channels=256),
                HiPLayer(num_groups=16, num_attn_layers=1, num_latents=128,
                         num_channels=64, num_heads=4, input_channels=128),
                HiPLayer(num_groups=64, num_attn_layers=1, num_latents=64,
                         num_channels=32, num_heads=2, input_channels=64),
                HiPLayer(num_groups=256, num_attn_layers=1, num_latents=32,
                         num_channels=16, num_heads=1, input_channels=32)
            ]
        )

    def forward(self, x):
        # Skip connection is not yet implemented for HiP-256 because the shape
        # doesn't match according to paper setup.
        for _, layer in enumerate(self.encoder):
            x = layer(x)[0]
        for _, layer in enumerate(self.decoder):
            x = layer(x)[0]
        return x


class HiPIO(nn.Module):
    def __init__(self, preprocess_layer, backbone, postprocess_layer):
        super().__init__()
        self.preprocess_layer = preprocess_layer
        self.backbone = backbone
        self.postprocess_layer = postprocess_layer

    def forward(self, x):
        x = self.preprocess_layer(x)
        x = self.backbone(x)
        x = self.postprocess_layer(x)
        return x



if __name__ == "__main__":

    hip16s = HiP16_small()
    hip16 = HiP16()
    hip256 = HiP256()

    x = torch.randn(1, 1024, 32)
    y = hip16s(x)
    print(y.shape)

    x = torch.randn(1, 1024, 32)
    y = hip16(x)
    print(y.shape)

    x = torch.randn(1, 1024, 16)
    y = hip256(x)
    print(y.shape)

    model = HiPIO(preprocess_layer, hip16s, postprocess_layer)

    # x = torch.randn(2, 3, 32, 32)
    x = torch.randn(2, 3, 16, 16)
    y = model(x)
    print(y.shape)
    imshow(y[0], batched=False)
