import re
import torch
import torch.nn as nn

from copy import deepcopy
from torch import Tensor
from torch.nn import Module, Linear, init
from typing import Any, Mapping, List

from diffusion.model.builder import MODELS
from diffusion.model.nets import PixArtMSBlock, PixArtMS, PixArt
from diffusion.model.nets.PixArt_blocks import TimestepEmbedder
from diffusion.model.nets.modulation_net import ModulationBlock
from diffusion.model.utils import auto_grad_checkpoint

@MODELS.register_module()
class MultiModalNet(Module):
    def __init__(
        self,
        token_size=32,
        in_channels=160,
        hidden_size=1152,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path: float = 0.,
        qk_norm=False,
        kv_compress_config=None,
        dtype=torch.float32,
        **kwargs,
    ) -> None:
        super().__init__()
        block_out_channels = [8, 16, 32, 64, 128]
        self.embedding = nn.ModuleList([nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1)])
        self.embedding.append(nn.GroupNorm(2, 8))
        self.embedding.append(nn.ReLU())
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.embedding.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.embedding.append(nn.GroupNorm(2, channel_in))
            self.embedding.append(nn.ReLU())
            self.embedding.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
            self.embedding.append(nn.GroupNorm(2, channel_out))
            self.embedding.append(nn.ReLU())
        self.embedding = nn.Sequential(*self.embedding)

        self.body_proj = nn.Linear(15, 32)
        self.in_channels = in_channels
        self.token_size = token_size
        self.num_heads = num_heads
        self.depth = 1
        self.hidden_size = hidden_size

        self.register_buffer("pos_embed", torch.zeros(1, token_size, hidden_size))
        self.condition_embedder = nn.Linear(self.in_channels, hidden_size) # (n, l, c) to (n, l, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        drop_path = [x.item() for x in torch.linspace(0, drop_path, self.depth)]  # stochastic depth decay rule
        self.kv_compress_config = kv_compress_config
        if kv_compress_config is None:
            self.kv_compress_config = {
                'sampling': None,
                'scale_factor': 1,
                'kv_compress_layer': [],
            }
        self.blocks = nn.ModuleList([
            ModulationBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[0],
                sampling=self.kv_compress_config['sampling'],
                sr_ratio=int(
                    self.kv_compress_config['scale_factor']
                ) if 0 in self.kv_compress_config['kv_compress_layer'] else 1,
                qk_norm=qk_norm,
            )
        ])
        self.dtype = dtype
        self.initialize_weights()


    def forward(self, c, timestep):
        
        sketch = c['sketch']
        body_params = c['body_params']
        timestep = timestep.to(self.dtype)
        sketch = self.embedding(sketch).reshape(sketch.shape[0], -1, 128)
        body_params = self.body_proj(body_params).repeat(1, sketch.shape[1], 1)
        condition = torch.cat([sketch, body_params], dim=-1)
        t = self.t_embedder(timestep.to(self.dtype))  # (N, D)
        t0 = self.t_block(t)

        x = self.condition_embedder(condition) + self.pos_embed.to(self.dtype)
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, t0)  # (N, T, D) #support grad checkpoint

        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos = torch.arange(0, self.token_size)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        hidden_size = self.pos_embed.shape[-1]
        _2i = torch.arange(0, hidden_size, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.pos_embed[..., 0::2] = torch.sin(pos / (10000 ** (_2i / hidden_size)))
        self.pos_embed[..., 1::2] = torch.cos(pos / (10000 ** (_2i / hidden_size)))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

class SewingLDM(Module):
    # only support single res model
    def __init__(self, base_model: PixArt, multimodal: MultiModalNet, control_scale: float = 1.0, idx: int = 0) -> None:
        super().__init__()
        self.base_model = base_model
        self.controlnet = multimodal
        self.scale = control_scale
        self.idx = idx

    def forward(self, x, timestep, y, mask=None, data_info=None, c=None, **kwargs):
        # modify the original PixArtMS forward function
        c = self.controlnet(c, timestep)
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        pos_embed = self.base_model.pos_embed.to(self.dtype)
        x = self.base_model.token_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.base_model.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.base_model.t_block(t)
        y = self.base_model.y_embedder(y, self.training)  # (N, 1, L, D)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1, 1, 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        for idx, block in enumerate(self.base_model.blocks):
            x = auto_grad_checkpoint(block, x, y, t0, y_lens)  # (N, T, D) #support grad checkpoint
            if idx == self.idx and c is not None:
                conditional_controls = c
                conditional_controls=nn.functional.adaptive_avg_pool1d(conditional_controls, x.shape[-1:])
                conditional_controls = conditional_controls.to(x)                
                mean_latents, std_latents = torch.mean(x, dim=(1, 2), keepdim=True), torch.std(x, dim=(1, 2), keepdim=True)
                mean_control, std_control = torch.mean(conditional_controls, dim=(1, 2), keepdim=True), torch.std(conditional_controls, dim=(1, 2), keepdim=True)
                conditional_controls = (conditional_controls - mean_control) * (std_latents / (std_control + 1e-12)) + mean_latents
                x = x + conditional_controls * self.scale

        x = self.base_model.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        return x

    def forward_with_dpmsolver(self, x, t, y, data_info, c, **kwargs):
        model_out = self.forward(x, t, y, data_info=data_info, c=c, **kwargs)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, data_info=None, c=None,  **kwargs):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask, data_info, c, **kwargs)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out.chunk(2, dim=2)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=2)

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype

