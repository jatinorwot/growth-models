import torch
import gc
import os

# Set memory optimization
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
  



########################### stable diffusion pipeline ############################

############################### code op ##############################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from collections import defaultdict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")





######################### data set loader, isme augmentation use nhi kr rahe ##############################
class MustardPlantDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=512, 
                 use_all_levels=True, specific_level=None, max_days=50):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        self.use_all_levels = use_all_levels
        self.specific_level = specific_level
        self.max_days = max_days
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        ################################# Validation transform (no augmentation) ###############################################
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.data = []
        self.load_mustard_dataset()
        
        print(f"\n✓ Loaded {len(self.data)} mustard plant images at {image_size}x{image_size} resolution")
        self.print_statistics()
    
    def load_mustard_dataset(self):
        mustard_dir = self.root_dir
        
        print(f"Loading mustard data from: {mustard_dir}")
        
        ####################### this will check p1, p2 , p3.... #########################
        plant_dirs = list(mustard_dir.glob("p*"))
        
        if plant_dirs:
            print(f"Found {len(plant_dirs)} plant directories")
            ################################ Structure: mustard/p1/d1/... or mustard/p1/d1/L1/... #########################################
            for plant_dir in sorted(plant_dirs):
                if not plant_dir.is_dir():
                    continue
                
                plant_id = plant_dir.name
                print(f"Processing plant: {plant_id}")
                
                for day_dir in sorted(plant_dir.glob("d*")):
                    if not day_dir.is_dir():
                        continue
                    
                    try:
                        day = int(day_dir.name[1:])  # Extract day number
                    except ValueError:
                        print(f"Skipping invalid directory: {day_dir.name}")
                        continue
                    
                    if day > self.max_days:
                        continue
                    
                    self._load_day_data(day_dir, day, plant_id)
        else:
            day_dirs = list(mustard_dir.glob("d*"))
            if day_dirs:
                print(f"Found {len(day_dirs)} day directories (no plant subdirs)")
                for day_dir in sorted(day_dirs):
                    if not day_dir.is_dir():
                        continue
                    
                    try:
                        day = int(day_dir.name[1:])
                    except ValueError:
                        continue
                    
                    if day > self.max_days:
                        continue
                    
                    self._load_day_data(day_dir, day, "p1")
            else:
                raise ValueError(f"No plant directories (p1, p2, ...) or day directories (d1, d2, ...) found in {mustard_dir}")
    
    def _load_day_data(self, day_dir, day, plant_id):
        # Check if images are directly in day folder
        direct_images = list(day_dir.glob("*.jpg")) + list(day_dir.glob("*.png"))
        
        if direct_images:
            # Images directly in day folder
            for img_path in direct_images:
                self.data.append({
                    'path': str(img_path),
                    'plant': plant_id,
                    'day': day,
                    'level': 'direct',
                    'age_normalized': day / self.max_days
                })
        else:
            levels = ['L1', 'L2', 'L3', 'L4', 'L5']
            if not self.use_all_levels and self.specific_level:
                levels = [self.specific_level]
            
            for level in levels:
                level_dir = day_dir / level
                if not level_dir.exists():
                    continue
                
                for ext in ['*.jpg', '*.png', '*.JPG', '*.PNG']: ################# just robust image hadling ##################
                    for img_path in level_dir.glob(ext):
                        self.data.append({
                            'path': str(img_path),
                            'plant': plant_id,
                            'day': day,
                            'level': level,
                            'age_normalized': day / self.max_days
                        })
    
    def print_statistics(self):
        if not self.data:
            print("\nNo mustard data loaded!")
            return
            
        stats = defaultdict(lambda: defaultdict(int))
        
        for item in self.data:
            stats['plants'][item['plant']] += 1
            stats['days'][item['day']] += 1
            stats['levels'][item['level']] += 1
        
        print("\nMustard Dataset Statistics:")
        print(f"├─ Total images: {len(self.data)}")
        print(f"├─ Number of plants: {len(stats['plants'])} ({', '.join(sorted(stats['plants'].keys()))})")
        print(f"├─ Number of days: {len(stats['days'])}")
        if stats['days']:
            print(f"├─ Days range: Day {min(stats['days'].keys())} - Day {max(stats['days'].keys())}")
        print(f"├─ Camera levels: {list(stats['levels'].keys())}")
        print(f"└─ Images per level: {dict(stats['levels'])}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image with high quality
        image = Image.open(item['path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'day': torch.tensor(item['day'], dtype=torch.long),
            'age_normalized': torch.tensor(item['age_normalized'], dtype=torch.float32),
            'metadata': {
                'level': item['level'],
                'path': item['path']
            }
        }

#################################  attention module for better quality ##############################
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
        x_flat = x.view(b, c, h*w).permute(0, 2, 1)  # [B, HW, C]
        
        x_norm = self.norm(x).view(b, c, h*w).permute(0, 2, 1)
        
        # Compute Q, K, V
        q = self.q(x_norm).view(b, h*w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x_norm).view(b, h*w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x_norm).view(b, h*w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, h*w, c)
        out = self.proj(out)
        
        # Add residual
        out = out + x_flat
        
        return out.permute(0, 2, 1).view(b, c, h, w)

############################################ High quality ResBlock with attention ##################################################
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
        
        # Add conditioning with scale and shift
        cond_proj = self.cond_proj(cond)
        scale, shift = torch.chunk(cond_proj, 2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        
        if self.use_attention:
            h = self.attention(h)
        
        h = F.silu(h)
        
        return h + self.shortcut(x)

########################### c UNet for high quality generation , memory efficient walaaaa ####################################
class MustardUNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=128, channel_mults=(1, 2, 3, 4), 
                 time_emb_dim=256, age_emb_dim=128, num_res_blocks=2):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Age embedding for growth stages
        self.age_embedding = nn.Sequential(
            nn.Linear(1, age_emb_dim),
            nn.SiLU(),
            nn.Linear(age_emb_dim, age_emb_dim),
            nn.SiLU(),
            nn.Linear(age_emb_dim, age_emb_dim)
        )
        
        # Combined conditioning
        self.cond_dim = time_emb_dim + age_emb_dim
        
        # Initial conv
        self.init_conv = nn.Conv2d(img_channels, base_channels, 7, padding=3)
        
        # Encoder
        self.downs = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            
            # Multiple ResBlocks per level
            for j in range(num_res_blocks):
                # Add attention only in the last stage to save memory
                use_attn = (i == len(channel_mults) - 1) and (j == num_res_blocks - 1)
                self.downs.append(
                    HighQualityResBlock(now_channels, out_channels, self.cond_dim, use_attn)
                )
                now_channels = out_channels
                channels.append(now_channels)
            
            # Downsample
            if i < len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)
        
        # Middle blocks - only one attention block to save memory
        self.mid_block1 = HighQualityResBlock(now_channels, now_channels, self.cond_dim, False)
        self.mid_block2 = HighQualityResBlock(now_channels, now_channels, self.cond_dim, True)
        
        # Decoder
        self.ups = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            out_channels = base_channels * mult
            
            for j in range(num_res_blocks + 1):
                # Minimal attention to save memory
                use_attn = False  # No attention in decoder
                self.ups.append(
                    HighQualityResBlock(
                        channels.pop() + now_channels, out_channels, self.cond_dim, use_attn
                    )
                )
                now_channels = out_channels
            
            # Upsample
            if i < len(channel_mults) - 1:
                self.ups.append(Upsample(now_channels))
        
        # Final layers
        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_conv = nn.Conv2d(base_channels, img_channels, 7, padding=3)
    
    def forward(self, x, t, age):
        # Get embeddings
        t_emb = self.time_mlp(t)
        age_emb = self.age_embedding(age.unsqueeze(1))
        
        # Combine conditions
        cond = torch.cat([t_emb, age_emb], dim=1)
        
        # Initial conv
        x = self.init_conv(x)
        
        # Encoder
        skips = [x]
        for layer in self.downs:
            if isinstance(layer, HighQualityResBlock):
                x = layer(x, cond)
            else:
                x = layer(x)
            skips.append(x)
        
        # Middle
        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)
        
        # Decoder
        for layer in self.ups:
            if isinstance(layer, HighQualityResBlock):
                x = torch.cat([x, skips.pop()], dim=1)
                x = layer(x, cond)
            else:
                x = layer(x)
        
        # Final layers
        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)
        
        return x

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

class MustardDiffusion:
    """diffusion model , high quality output """
    def __init__(self, img_size=512, device=device, train_steps=1000):
        self.img_size = img_size
        self.device = device
        self.train_steps = train_steps
        
        # Initialize mustard-specific UNet
        self.unet = MustardUNet().to(device)
        
        # Cosine noise schedule for better quality
        self.beta_start = 0.00085
        self.beta_end = 0.012
        
        # Cosine schedule
        steps = torch.linspace(0, train_steps - 1, train_steps, dtype=torch.float32)
        alpha_bar = torch.cos((steps / train_steps + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        self.betas = torch.cat([torch.tensor([0.0]), betas]).clip(0, 0.999).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Optimizer with lower learning rate for stability
        self.optimizer = AdamW(self.unet.parameters(), lr=5e-5, weight_decay=0.01)
        
        # EMA for high-quality generation
        self.ema_rate = 0.9999
        self.ema_model = None
        
        # Training metrics
        self.losses = []
    
    def get_noise_schedule(self, t, x):
        """Get noise schedule parameters"""
        alpha_bar = self.alpha_bars[t]
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        return alpha_bar
    
    def forward_diffusion(self, x0, t):
        """Add noise to images"""
        noise = torch.randn_like(x0)
        alpha_bar = self.get_noise_schedule(t, x0)
        
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return xt, noise
    
    def train_step(self, batch):
        """Single training step"""
        images = batch['image'].to(self.device)
        age = batch['age_normalized'].to(self.device)
        batch_size = images.shape[0]
        
        # Random timesteps
        t = torch.randint(0, self.train_steps, (batch_size,)).to(self.device)
        
        # Forward diffusion
        xt, noise = self.forward_diffusion(images, t)
        
        # Predict noise
        predicted_noise = self.unet(xt, t, age)
        
        # Calculate loss with variance reduction
        loss = F.mse_loss(predicted_noise, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        self.optimizer.step()
        
        # Update EMA
        if self.ema_model is not None:
            self.update_ema()
        
        return loss.item()
    
    def update_ema(self):
        """Update exponential moving average"""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.unet.parameters()):
                ema_param.data.mul_(self.ema_rate).add_(param.data, alpha=1 - self.ema_rate)
    
    @torch.no_grad()
    def sample_ddim(self, age, num_samples=1, steps=50, eta=0.0):
        """DDIM sampling for faster high-quality generation"""
        model = self.ema_model if self.ema_model is not None else self.unet
        model.eval()
        
        # Start from random noise
        x = torch.randn(num_samples, 3, self.img_size, self.img_size).to(self.device)
        
        # Prepare age conditioning
        if isinstance(age, (int, float)):
            age = torch.tensor([age / 50.0] * num_samples).to(self.device)
        
        # DDIM sampling with fewer steps
        time_steps = torch.linspace(self.train_steps - 1, 0, steps + 1).long()
        
        for i in tqdm(range(steps), desc='Generating mustard images', leave=False):
            t = time_steps[i].to(self.device)
            t_next = time_steps[i + 1].to(self.device)
            
            t_batch = torch.full((num_samples,), t, device=self.device)
            
            # Predict noise
            predicted_noise = model(x, t_batch, age)
            
            # Get alphas
            alpha_t = self.alpha_bars[t]
            alpha_t_next = self.alpha_bars[t_next] if t_next >= 0 else torch.tensor(1.0)
            
            # DDIM update
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_next - eta**2 * (1 - alpha_t_next) / (1 - alpha_t)) * predicted_noise
            
            # Random noise
            noise = eta * torch.randn_like(x) * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t))
            
            # Next sample
            x = torch.sqrt(alpha_t_next) * pred_x0 + dir_xt + noise
        
        model.train()
        return x
    
    def save_checkpoint(self, epoch, path, save_ema=True):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
        }
        
        if save_ema and self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
        
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.unet.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.losses = checkpoint.get('losses', [])
        
        if 'ema_model_state_dict' in checkpoint:
            if self.ema_model is None:
                self.ema_model = MustardUNet().to(self.device)
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        
        print(f"✓ Checkpoint loaded: {path}")
        return checkpoint['epoch']

def train_mustard_model(data_dir, num_epochs=150, batch_size=8, 
                       image_size=512, save_dir='mustard_checkpoints'):
    """Train high-quality plant diffusion model"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/samples", exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"High-Quality Mustard Plant Diffusion Training")
    print(f"{'='*50}")
    print(f"Data Directory: {data_dir}")
    print(f"Image Size: {image_size}x{image_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}")
    print(f"{'='*50}\n")
    
    # Data preparation
    dataset = MustardPlantDataset(
        root_dir=data_dir,
        image_size=image_size,
        use_all_levels=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True
    )
    
    # Initialize model
    model = MustardDiffusion(img_size=image_size)
    
    # Initialize EMA model
    model.ema_model = MustardUNet().to(device)
    model.ema_model.load_state_dict(model.unet.state_dict())
    
    # Calculate total steps for scheduler
    total_steps = len(dataloader) * num_epochs
    scheduler = OneCycleLR(
        model.optimizer, 
        max_lr=1e-4, 
        total_steps=total_steps,
        pct_start=0.1
    )
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            loss = model.train_step(batch)
            epoch_loss += loss
            scheduler.step()
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({'loss': f'{loss:.4f}', 'lr': f'{current_lr:.6f}'})
        
        # Average loss
        avg_loss = epoch_loss / len(dataloader)
        model.losses.append(avg_loss)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_checkpoint(epoch+1, f"{save_dir}/best_mustard_model.pth")
            print(f"✓ New best model! Loss: {best_loss:.4f}")
        
        # Generate samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("\nGenerating mustard plant samples...")
            
            # Sample ages throughout mustard growth cycle
            sample_ages = [1, 10, 20, 30, 40, 50]
            samples = []
            
            for age in sample_ages:
                # Use DDIM for faster sampling
                sample = model.sample_ddim(age, num_samples=1, steps=50)
                samples.append(sample)
            
            # Stack and denormalize
            samples = torch.cat(samples, dim=0)
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            
            # Save high-quality samples
            save_image(samples, f"{save_dir}/samples/mustard_epoch_{epoch+1}.png", 
                      nrow=len(sample_ages), padding=4)
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                model.save_checkpoint(epoch+1, f"{save_dir}/mustard_checkpoint_epoch_{epoch+1}.pth")
    
    # Save final model
    model.save_checkpoint(num_epochs, f"{save_dir}/final_mustard_model.pth")
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(model.losses)
    plt.title('Mustard Plant Model Training Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/mustard_training_history.png", dpi=150)
    plt.close()
    
    print(f"\n{'='*50}")
    print(f"Mustard Model Training Complete!")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Models saved in: {save_dir}")
    print(f"{'='*50}\n")
    
    return model

def generate_mustard_images(model_path, age, num_images=4, 
                           output_dir='generated_mustard', image_size=512):
    """Generate high-quality mustard plant images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = MustardDiffusion(img_size=image_size)
    epoch = model.load_checkpoint(model_path)
    
    print(f"\n{'='*50}")
    print(f"Generating High-Quality Mustard Plant Images")
    print(f"{'='*50}")
    print(f"Model: {model_path} (epoch {epoch})")
    print(f"Age: Day {age}")
    print(f"Number of images: {num_images}")
    print(f"Resolution: {image_size}x{image_size}")
    print(f"{'='*50}\n")
    
    # Generate images using DDIM for quality
    images = model.sample_ddim(age, num_samples=num_images, steps=100)
    
    # Denormalize
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    # Save individual high-quality images
    for i, img in enumerate(images):
        save_path = f"{output_dir}/mustard_day_{age}_hq_{i+1}.png"
        save_image(img, save_path)
        print(f"✓ Saved: {save_path}")
    
    # Save grid
    grid_path = f"{output_dir}/mustard_day_{age}_grid.png"
    save_image(images, grid_path, nrow=min(4, num_images), padding=4)
    print(f"✓ Saved grid: {grid_path}")
    
    print(f"\n✓ Generation complete! Mustard images saved to: {output_dir}\n")
    
    return images

def generate_mustard_growth_timeline(model_path, start_day=1, end_day=50, 
                                   num_samples=10, output_dir='mustard_timeline',
                                   image_size=512):
    """Generate mustard plant growth timeline"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = MustardDiffusion(img_size=image_size)
    model.load_checkpoint(model_path)
    
    # Select days to visualize
    days = np.linspace(start_day, end_day, num_samples, dtype=int)
    
    print(f"\n{'='*50}")
    print(f"Generating Mustard Growth Timeline")
    print(f"{'='*50}")
    print(f"Days: {days}")
    print(f"{'='*50}\n")
    
    samples = []
    for day in tqdm(days, desc="Generating mustard timeline"):
        sample = model.sample_ddim(day, num_samples=1, steps=50)
        samples.append(sample)
    
    # Stack and denormalize
    samples = torch.cat(samples, dim=0)
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # Create figure with labels
    fig, axes = plt.subplots(1, num_samples, figsize=(3*num_samples, 4))
    
    for i, (img, day) in enumerate(zip(samples, days)):
        img_np = img.cpu().permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].set_title(f'Day {day}')
        axes[i].axis('off')
    
    plt.suptitle('Mustard Plant Growth Timeline', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mustard_growth_timeline.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save as grid
    save_image(samples, f"{output_dir}/mustard_timeline_grid.png", 
              nrow=num_samples, padding=2)
    
    print(f"✓ Mustard growth timeline saved to {output_dir}")


# if __name__ == "__main__":
#     mustard_path = '/home/jovyan/Shreya/Jatin/mustard/content/drive/MyDrive/ACM grand challenge/Crops data/For_age_prediction/mustard'
    
#     # Train mustard model
#     model = train_mustard_model(
#         data_dir=mustard_path,
#         num_epochs=200,
#         batch_size=16,     
#         image_size=256,   
#         save_dir='mustard_checkpoints'
#      )

# def generate_single_plant_timeline_grid(seed=42):
#     """Generate same plant from day 1 to 50 in a single image"""
    
#     import os
#     os.makedirs('okra_single_plant', exist_ok=True)
    
#     # Load model
#     model = MustardDiffusion(img_size=256)
#     model.load_checkpoint(r'C:\Users\mrigm\Downloads\best_radish_model (1).pth')

    
#     # Days from 1 to 50 with 5-day intervals
#     days = list(range(0, 50, 5))  # [1, 6, 11, 16, 21, 26, 31, 36, 41, 46]
#     print(f"Generating plant (seed {seed}) for days: {days}")
    
#     all_images = []
    
#     for day in days:
#         print(f"Generating day {day}...")
        
#         # Reset to same seed for consistent plant
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         np.random.seed(seed)
        
#         # Generate with fixed seed
#         with torch.no_grad():
#             image = model.sample_ddim(
#                 age=day,
#                 num_samples=1,
#                 steps=100,  # High quality
#                 eta=0.0  # Deterministic
#             )
        
#         all_images.append(image)
    
#     # Stack all images
#     all_images = torch.cat(all_images, dim=0)
#     all_images = (all_images + 1) / 2
#     all_images = torch.clamp(all_images, 0, 1)
    
#     # Save as single grid image
#     save_image(all_images, f"wheat{seed}_day1to50_new.png", 
#               nrow=5, padding=4)  # 5 images per row for 10 total images
    
#     # Create labeled version with matplotlib
#     import matplotlib.pyplot as plt
    
#     fig, axes = plt.subplots(2, 5, figsize=(15, 6))
#     axes = axes.flatten()
    
#     for i, (img, day) in enumerate(zip(all_images, days)):
#         img_np = img.cpu().permute(1, 2, 0).numpy()
#         axes[i].imshow(img_np)
#         axes[i].set_title(f'Day {day}', fontsize=11)
#         axes[i].axis('off')
    
#     plt.suptitle(f'Okra Plant Growth Timeline - Same Plant (Seed {seed})', fontsize=14)
#     plt.tight_layout()
#     plt.savefig(f"radish_plant_seed{seed}_day1to50_labeled_new.png", dpi=200, bbox_inches='tight')
#     plt.close()
    
#     print(f"✓ Generated single image with plant timeline")
#     print(f"✓ Saved as: okra_plant_seed{seed}_day1to50.png")
#     print(f"✓ Labeled version: okra_plant_seed{seed}_day1to50_labeled.png")

# # Generate the timeline
# generate_single_plant_timeline_grid(seed=42)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import gradio as gr
import shutil
import tempfile

# --- Your diffusion model class should be imported or defined above ---
# from your_model_file import MustardDiffusion


# def generate_single_plant_timeline_grid(seed=42, crop_type="okra"):
#     """Generate same plant from day 1 to 50 in a single image"""
    
#     os.makedirs(f'{crop_type}_single_plant', exist_ok=True)
#    # "C:\Users\mrigm\Downloads\best_mustard_model.pth"
#     # Load model
#     model = MustardDiffusion(img_size=256)
#     model.load_checkpoint(fr'C:\Users\mrigm\Downloads\best_{crop_type}_model.pth')

    
#     # Days from 1 to 50 with 5-day intervals
#     days = list(range(0, 50, 5))
#     print(f"Generating plant (seed {seed}) for days: {days}")
    
#     all_images = []
    
#     for day in days:
#         print(f"Generating day {day}...")
        
#         # Reset seed for consistent plant
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         np.random.seed(seed)
        
#         # Generate image
#         with torch.no_grad():
#             image = model.sample_ddim(
#                 age=day,
#                 num_samples=1,
#                 steps=100,
#                 eta=0.0
#             )
#         all_images.append(image)
    
#     # Stack and normalize
#     all_images = torch.cat(all_images, dim=0)
#     all_images = (all_images + 1) / 2
#     all_images = torch.clamp(all_images, 0, 1)
    
#     # Save unlabeled grid
#     raw_path = f"{crop_type}_single_plant/{crop_type}_seed{seed}_timeline.png"
#     save_image(all_images, raw_path, nrow=5, padding=4)
    
#     # Create labeled version
#     fig, axes = plt.subplots(2, 5, figsize=(15, 6))
#     axes = axes.flatten()
    
#     for i, (img, day) in enumerate(zip(all_images, days)):
#         img_np = img.cpu().permute(1, 2, 0).numpy()
#         axes[i].imshow(img_np)
#         axes[i].set_title(f'Day {day}', fontsize=11)
#         axes[i].axis('off')
    
#     plt.suptitle(f'{crop_type.capitalize()} Plant Growth Timeline (Seed {seed})', fontsize=14)
#     plt.tight_layout()
    
#     labeled_path = os.path.abspath(f"{crop_type}_single_plant/{crop_type}_seed{seed}_timeline_labeled.png")
#     plt.savefig(labeled_path, dpi=200, bbox_inches='tight')
#     plt.close()
    
#     print(f"✓ Generated single image with plant timeline: {labeled_path}")
#     return labeled_path

def generate_single_plant_timeline_grid(
    seed=None, 
    crop_type="okra", 
    start_day=0, 
    end_day=50, 
    interval=5
):
    """Generate plant growth timeline with flexible days and seed"""
    
    os.makedirs(f'{crop_type}_single_plant', exist_ok=True)
    
    # Choose seed
    if seed is None or seed == "random":
        seed = np.random.randint(0, 100000)
    
    # Load model
    model = MustardDiffusion(img_size=256)
    model.load_checkpoint(fr'C:\Users\mrigm\Downloads\best_{crop_type}_model.pth')
    
    # Days range
    days = list(range(start_day, end_day + 1, interval))
    print(f"Generating plant (seed {seed}) for days: {days}")
    
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
    
    all_images = torch.cat(all_images, dim=0)
    all_images = (all_images + 1) / 2
    all_images = torch.clamp(all_images, 0, 1)
    
    # Save unlabeled grid
    raw_path = f"{crop_type}_single_plant/{crop_type}_seed{seed}_timeline.png"
    save_image(all_images, raw_path, nrow=5, padding=4)
    
    # Labeled version
    fig, axes = plt.subplots(
    nrows=int(np.ceil(len(days)/5)), 
    ncols=5,  # always 5 columns
    figsize=(15, 3 * int(np.ceil(len(days)/5)))
    )
    axes = axes.flatten()

    for i, (img, day) in enumerate(zip(all_images, days)):
        img_np = img.cpu().permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].set_title(f'Day {day}', fontsize=11)
        axes[i].axis('off')

    # Hide any remaining empty axes
    for j in range(len(days), len(axes)):
        axes[j].axis('off')

    
    # for i, (img, day) in enumerate(zip(all_images, days)):
    #     img_np = img.cpu().permute(1, 2, 0).numpy()
    #     axes[i].imshow(img_np)
    #     axes[i].set_title(f'Day {day}', fontsize=11)
    #     axes[i].axis('off')
    
    plt.suptitle(f'{crop_type.capitalize()} Plant Growth Timeline (Seed {seed})', fontsize=14)
    plt.tight_layout()
    
    labeled_path = os.path.abspath(f"{crop_type}_single_plant/{crop_type}_seed{seed}_timeline_labeled.png")
    plt.savefig(labeled_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated single image with plant timeline: {labeled_path}")
    return labeled_path



# --- Gradio wrapper function ---
with gr.Blocks(theme=gr.themes.Soft(
    
)) as demo:
    gr.Markdown(
        """
        # 🌿 Plant Growth Generator
        Generate realistic plant growth timelines using diffusion models.
        """
    )

    with gr.Row():
        seed = gr.Dropdown(
            ["Random", "42", "123", "999"], 
            label="Seed", 
            value="42",
            info="Choose a fixed seed or Random"
        )
        crop_type = gr.Dropdown(
            ["okra", "wheat", "radish", "mustard"],
            label="Crop Type",
            value="okra"
        )

    with gr.Row():
        start_day = gr.Number(label="Start Day", value=0)
        end_day = gr.Number(label="End Day", value=20)
        interval = gr.Number(label="Interval (days)", value=2)

    generate_btn = gr.Button("⚡ Generate Timeline", variant="primary")

    output_image = gr.Image(label="Generated Plant Timeline", type="filepath")
    status_text = gr.Textbox(label="Status", interactive=False)

    # def gradio_wrapper(seed_choice, crop_type, start, end, interval):

    #     # Convert Random to None
    #     seed_value = None if seed_choice == "Random" else int(seed_choice)
    #     return generate_single_plant_timeline_grid(seed=seed_value, crop_type=crop_type,
    #                                                start_day=int(start), end_day=int(end), interval=int(interval))
    
    def gradio_wrapper(seed_choice, crop_type, start, end, interval):
        try:
            # Convert Random to None
            seed_value = None if seed_choice == "Random" else int(seed_choice)
            
            # Generate timeline
            result_path = generate_single_plant_timeline_grid(
                seed=seed_value, 
                crop_type=crop_type,
                start_day=int(start), 
                end_day=int(end), 
                interval=int(interval)
            )
            
            # Return both image path and status message
            return result_path, f"✅ Successfully generated {crop_type} plant timeline (Seed {seed_value})"
        
        except Exception as e:
            return None, f"❌ Error: {str(e)}"


    generate_btn.click(fn=gradio_wrapper, 
                       inputs=[seed, crop_type, start_day, end_day, interval], 
                       outputs=[output_image, status_text])

    gr.Markdown("Made with ❤️ using diffusion models")

demo.launch(share=True)

