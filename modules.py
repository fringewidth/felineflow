import os
import math
import torch
import torch.nn as nn

class Config:
    IMAGE_SIZE = 128
    BATCH_SIZE = 16
    EPOCHS = 200
    LR = 2e-4
    TIMESTEPS = 1000
    BETA_START = 1e-4
    BETA_END = 0.02
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    DATASET_PATH = "dataset"
    ZIP_PATH = "cats_processed.zip"
    ZIP_URL = "https://github.com/fringewidth/felineflow/raw/main/cats_processed.zip"
    MODEL_DIR = "models"
    SAMPLE_DIR = "samples"
    GPU_MEMORY_RATIO = 0.3

class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps, size=(n,))


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        h = h + time_emb[:, :, None, None]
        h = self.bnorm2(self.relu(self.conv2(h)))
        h = self.dropout(h)
        return self.transform(h)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.to_qkv = nn.Linear(channels, channels * 3, bias=False)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).swapaxes(1, 2)
        x_ln = self.ln(x)
        qkv = self.to_qkv(x_ln).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, h*w, 4, c//4).transpose(1, 2), qkv)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, h*w, c)
        out = self.to_out(out)
        out = out + x
        out = self.ff_self(out) + out
        return out.swapaxes(2, 1).view(b, c, h, w)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 3

        down_channels = (64, 128, 256, 512)
        up_channels = (512, 256, 128, 64)
        out_dim = 3
        time_dim = 128

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_dim, dropout=0.1) \
            for i in range(len(down_channels)-1)
        ])
        
        self.sa1 = AttentionBlock(down_channels[2]) # 32x32 -> 1024 tokens
        self.sa2 = AttentionBlock(down_channels[3]) # 16x16 -> 256 tokens

        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], time_dim, up=True, dropout=0.1) \
            for i in range(len(up_channels)-1)
        ])
        
        self.sa3 = AttentionBlock(up_channels[1]) # 32x32
        self.sa4 = AttentionBlock(up_channels[2]) # 64x64

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.conv0(x)
        
        residual_inputs = []
        for i, down in enumerate(self.downs):
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(down), x, t, use_reentrant=False)
            
            if i == 1: x = self.sa1(x)
            if i == 2: x = self.sa2(x)
            residual_inputs.append(x)
            
        for i, up in enumerate(self.ups):
            res_x = residual_inputs.pop()
            x = torch.cat((x, res_x), dim=1)
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(up), x, t, use_reentrant=False)
            if i == 0: x = self.sa3(x)
            if i == 1: x = self.sa4(x)
            
        return self.output(x)

