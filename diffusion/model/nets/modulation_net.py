import re
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from diffusion.model.nets.PixArt_blocks import t2i_modulate, AttentionKVCompress, T2IFinalLayer, TimestepEmbedder
from diffusion.model.utils import auto_grad_checkpoint

class ModulationBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0,
                 sampling=None, sr_ratio=1, qk_norm=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionKVCompress(
            hidden_size, num_heads=num_heads, qkv_bias=True, sampling=sampling, sr_ratio=sr_ratio,
            qk_norm=qk_norm, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
        self.sampling = sampling
        self.sr_ratio = sr_ratio

    def forward(self, x, t, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


class ModulationNet(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            token_size=32,
            in_channels=64,
            out_channels=64,
            hidden_size=512,
            depth=8,
            num_heads=6,
            mlp_ratio=4.0,
            drop_path: float = 0.,
            qk_norm=False,
            kv_compress_config=None,
            **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.token_size = token_size
        self.num_heads = num_heads
        self.depth = depth
        self.hidden_size = hidden_size

        self.register_buffer("pos_embed", torch.zeros(1, token_size, hidden_size))
        self.token_embedder = nn.Linear(self.in_channels, hidden_size) # (n, l, c) to (n, l, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.kv_compress_config = kv_compress_config
        if kv_compress_config is None:
            self.kv_compress_config = {
                'sampling': None,
                'scale_factor': 1,
                'kv_compress_layer': [],
            }
        self.blocks = nn.ModuleList([
            ModulationBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                sampling=self.kv_compress_config['sampling'],
                sr_ratio=int(
                    self.kv_compress_config['scale_factor']
                ) if i in self.kv_compress_config['kv_compress_layer'] else 1,
                qk_norm=qk_norm,
            )
            for i in range(depth)
        ])
        self.final_layer = T2IFinalLayer(hidden_size, self.out_channels)

        self.initialize_weights()


    def forward(self, x, timestep, data_info=None, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, L, C) tensor of cloth token inputs 
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        x = self.token_embedder(x) + pos_embed # (N, L, D)
        # pos_embed = self.pos_embed.to(self.dtype)
        # self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        # x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, t0)  # (N, T, D) #support grad checkpoint
        x = self.final_layer(x, t)  # (N, T, out_channels)
        # x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x.chunk(2, dim=2)


    def forward_with_cfg(self, x, timestep, cfg_scale, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, **kwargs)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

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

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

