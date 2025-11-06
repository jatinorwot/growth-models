import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from pathlib import Path
import math
from tqdm import tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class CrossAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h*w).permute(0, 2, 1)
        x_norm = self.norm(x).view(b, c, h*w).permute(0, 2, 1)
        
        q = self.q(x_norm).view(b, h*w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x_norm).view(b, h*w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x_norm).view(b, h*w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, h*w, c)
        out = self.proj(out)
        out = out + x_flat
        
        return out.permute(0, 2, 1).view(b, c, h, w)


class HighQualityResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2)
        )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if use_attention:
            self.attention = CrossAttention(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        cond_proj = self.cond_proj(cond)
        scale, shift = torch.chunk(cond_proj, 2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        
        if self.use_attention:
            h = self.attention(h)
        
        h = F.silu(h)
        
        return h + self.shortcut(x)


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


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)


class MustardUNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=128, channel_mults=(1, 2, 3, 4), 
                 time_emb_dim=256, age_emb_dim=128, num_res_blocks=2):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.age_embedding = nn.Sequential(
            nn.Linear(1, age_emb_dim),
            nn.SiLU(),
            nn.Linear(age_emb_dim, age_emb_dim),
            nn.SiLU(),
            nn.Linear(age_emb_dim, age_emb_dim)
        )
        
        self.cond_dim = time_emb_dim + age_emb_dim
        self.init_conv = nn.Conv2d(img_channels, base_channels, 7, padding=3)
        
        self.downs = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            
            for j in range(num_res_blocks):
                use_attn = (i == len(channel_mults) - 1) and (j == num_res_blocks - 1)
                self.downs.append(
                    HighQualityResBlock(now_channels, out_channels, self.cond_dim, use_attn)
                )
                now_channels = out_channels
                channels.append(now_channels)
            
            if i < len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)
        
        self.mid_block1 = HighQualityResBlock(now_channels, now_channels, self.cond_dim, False)
        self.mid_block2 = HighQualityResBlock(now_channels, now_channels, self.cond_dim, True)
        
        self.ups = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            out_channels = base_channels * mult
            
            for j in range(num_res_blocks + 1):
                self.ups.append(
                    HighQualityResBlock(
                        channels.pop() + now_channels, out_channels, self.cond_dim, False
                    )
                )
                now_channels = out_channels
            
            if i < len(channel_mults) - 1:
                self.ups.append(Upsample(now_channels))
        
        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_conv = nn.Conv2d(base_channels, img_channels, 7, padding=3)
    
    def forward(self, x, t, age):
        t_emb = self.time_mlp(t)
        age_emb = self.age_embedding(age.unsqueeze(1))
        cond = torch.cat([t_emb, age_emb], dim=1)
        
        x = self.init_conv(x)
        
        skips = [x]
        for layer in self.downs:
            if isinstance(layer, HighQualityResBlock):
                x = layer(x, cond)
            else:
                x = layer(x)
            skips.append(x)
        
        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)
        
        for layer in self.ups:
            if isinstance(layer, HighQualityResBlock):
                x = torch.cat([x, skips.pop()], dim=1)
                x = layer(x, cond)
            else:
                x = layer(x)
        
        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)
        
        return x


class PlantDiffusion:
    def __init__(self, img_size=512, device=device, train_steps=1000):
        self.img_size = img_size
        self.device = device
        self.train_steps = train_steps
        
        self.unet = MustardUNet().to(device)
        
        self.beta_start = 0.00085
        self.beta_end = 0.012
        
        steps = torch.linspace(0, train_steps - 1, train_steps, dtype=torch.float32)
        alpha_bar = torch.cos((steps / train_steps + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        self.betas = torch.cat([torch.tensor([0.0]), betas]).clip(0, 0.999).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        self.ema_model = None
    
    @torch.no_grad()
    def sample_ddim(self, age, num_samples=1, steps=50, eta=0.0):
        model = self.ema_model if self.ema_model is not None else self.unet
        model.eval()
        
        x = torch.randn(num_samples, 3, self.img_size, self.img_size).to(self.device)
        
        if isinstance(age, (int, float)):
            age = torch.tensor([age / 50.0] * num_samples).to(self.device)
        
        time_steps = torch.linspace(self.train_steps - 1, 0, steps + 1).long()
        
        for i in tqdm(range(steps), desc='Generating images', leave=False):
            t = time_steps[i].to(self.device)
            t_next = time_steps[i + 1].to(self.device)
            
            t_batch = torch.full((num_samples,), t, device=self.device)
            
            predicted_noise = model(x, t_batch, age)
            
            alpha_t = self.alpha_bars[t]
            alpha_t_next = self.alpha_bars[t_next] if t_next >= 0 else torch.tensor(1.0)
            
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            dir_xt = torch.sqrt(1 - alpha_t_next - eta**2 * (1 - alpha_t_next) / (1 - alpha_t)) * predicted_noise
            
            noise = eta * torch.randn_like(x) * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t))
            
            x = torch.sqrt(alpha_t_next) * pred_x0 + dir_xt + noise
        
        model.train()
        return x
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.unet.load_state_dict(checkpoint['model_state_dict'])
        
        if 'ema_model_state_dict' in checkpoint:
            if self.ema_model is None:
                self.ema_model = MustardUNet().to(self.device)
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        
        print(f"✓ Checkpoint loaded: {path}")
        return checkpoint['epoch']


# Model paths configuration
MODEL_PATHS = {
    'okra': 'best_okra_model.pth',
    'wheat': 'best_wheat_model.pth',
    'radish': 'best_radish_model.pth',
    'mustard': 'best_mustard_model.pth'
}


def generate_single_plant_timeline(
    crop_type="okra",
    seed=42, 
    start_day=0, 
    end_day=50, 
    interval=5,
    output_dir=None
):
    """Generate plant growth timeline for a specific crop type"""
    
    # Validate crop type
    if crop_type not in MODEL_PATHS:
        raise ValueError(f"Invalid crop type. Choose from: {list(MODEL_PATHS.keys())}")
    
    # Set output directory
    if output_dir is None:
        output_dir = f'{crop_type}_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle random seed
    if seed is None or seed == "random":
        seed = np.random.randint(0, 100000)
    
    print(f"\n{'='*60}")
    print(f"Generating {crop_type.upper()} Plant Growth Timeline")
    print(f"{'='*60}")
    print(f"Seed: {seed}")
    print(f"Days: {start_day} to {end_day} (interval: {interval})")
    print(f"Model: {MODEL_PATHS[crop_type]}")
    print(f"{'='*60}\n")
    
    # Load model
    model_path = MODEL_PATHS[crop_type]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = PlantDiffusion(img_size=256)
    model.load_checkpoint(model_path)
    
    # Generate timeline
    days = list(range(start_day, end_day + 1, interval))
    all_images = []
    
    for day in days:
        print(f"Generating day {day}...")
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        with torch.no_grad():
            image = model.sample_ddim(
                age=day,
                num_samples=1,
                steps=100,
                eta=0.0
            )
        all_images.append(image)
    
    # Process images
    all_images = torch.cat(all_images, dim=0)
    all_images = (all_images + 1) / 2
    all_images = torch.clamp(all_images, 0, 1)
    
    # Save raw grid
    raw_path = f"{output_dir}/{crop_type}_seed{seed}_timeline.png"
    save_image(all_images, raw_path, nrow=5, padding=4)
    
    # Create labeled version
    n_rows = int(np.ceil(len(days) / 5))
    fig, axes = plt.subplots(
        nrows=n_rows, 
        ncols=5,
        figsize=(15, 3 * n_rows)
    )
    axes = axes.flatten()

    for i, (img, day) in enumerate(zip(all_images, days)):
        img_np = img.cpu().permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].set_title(f'Day {day}', fontsize=11)
        axes[i].axis('off')

    # Hide empty subplots
    for j in range(len(days), len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f'{crop_type.capitalize()} Plant Growth Timeline (Seed {seed})', fontsize=14)
    plt.tight_layout()
    
    labeled_path = os.path.abspath(f"{output_dir}/{crop_type}_seed{seed}_timeline_labeled.png")
    plt.savefig(labeled_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Generated timeline saved to:")
    print(f"  - Raw grid: {raw_path}")
    print(f"  - Labeled: {labeled_path}\n")
    
    return labeled_path


def generate_all_crops_comparison(
    seed=42,
    day=25,
    output_dir='crops_comparison'
):
    """Generate comparison of all 4 crops at the same day"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating All Crops Comparison at Day {day}")
    print(f"{'='*60}\n")
    
    all_crop_images = []
    crop_names = []
    
    for crop_type in MODEL_PATHS.keys():
        print(f"Generating {crop_type}...")
        
        # Load model
        model = PlantDiffusion(img_size=256)
        model.load_checkpoint(MODEL_PATHS[crop_type])
        
        # Set seed for consistency
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate image
        with torch.no_grad():
            image = model.sample_ddim(age=day, num_samples=1, steps=100, eta=0.0)
        
        all_crop_images.append(image)
        crop_names.append(crop_type)
    
    # Stack all images
    all_crop_images = torch.cat(all_crop_images, dim=0)
    all_crop_images = (all_crop_images + 1) / 2
    all_crop_images = torch.clamp(all_crop_images, 0, 1)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, (img, name) in enumerate(zip(all_crop_images, crop_names)):
        img_np = img.cpu().permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].set_title(name.capitalize(), fontsize=14, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle(f'All Crops at Day {day} (Seed {seed})', fontsize=16)
    plt.tight_layout()
    
    comparison_path = f"{output_dir}/all_crops_day{day}_seed{seed}.png"
    plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Comparison saved to: {comparison_path}\n")
    return comparison_path


# Example usage
if __name__ == "__main__":
    # Generate timeline for individual crops
    for crop in ['okra', 'wheat', 'radish', 'mustard']:
        generate_single_plant_timeline(
            crop_type=crop,
            seed=42,
            start_day=0,
            end_day=50,
            interval=5
        )
    
    # Generate comparison of all crops at specific day
    generate_all_crops_comparison(seed=42, day=25)
