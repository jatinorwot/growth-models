# Plant Growth Timeline Generator 🌱

A deep learning-based plant growth visualization system using diffusion models to generate realistic plant images across different growth stages for **Okra, Wheat, Radish, and Mustard** crops.

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Model Files](#model-files)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Functions Reference](#functions-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** with **DDIM sampling** to generate high-quality plant growth images. The model is conditioned on:
- **Time step** (for the diffusion process)
- **Plant age** (day 0-50 of growth)

### Key Features
- ✅ Support for 4 crop types (Okra, Wheat, Radish, Mustard)
- ✅ High-quality 256x256 image generation
- ✅ Consistent plant generation using seeds
- ✅ Growth timeline visualization
- ✅ Cross-crop comparison at specific growth stages

---

## Architecture

### 1. **Model Components**

#### **MustardUNet** (Main Generator)
The U-Net architecture with the following components:
```
Input Image (3x256x256)
    ↓
Initial Conv (7x7) → 128 channels
    ↓
┌─────────────── Encoder ───────────────┐
│  Level 1: ResBlocks + Attention       │
│  Downsample (128 → 256 channels)      │
│  Level 2: ResBlocks                   │
│  Downsample (256 → 384 channels)      │
│  Level 3: ResBlocks                   │
│  Downsample (384 → 512 channels)      │
│  Level 4: ResBlocks + Attention       │
└───────────────────────────────────────┘
    ↓
┌──────────── Middle Blocks ────────────┐
│  Mid Block 1: ResBlock                │
│  Mid Block 2: ResBlock + Attention    │
└───────────────────────────────────────┘
    ↓
┌─────────────── Decoder ───────────────┐
│  Level 4: ResBlocks + Skip Connection │
│  Upsample (512 → 384 channels)        │
│  Level 3: ResBlocks + Skip Connection │
│  Upsample (384 → 256 channels)        │
│  Level 2: ResBlocks + Skip Connection │
│  Upsample (256 → 128 channels)        │
│  Level 1: ResBlocks + Skip Connection │
└───────────────────────────────────────┘
    ↓
Final Conv (7x7) → 3 channels
    ↓
Output Image (3x256x256)
```

**Conditioning:**
- **Time Embedding**: Sinusoidal positional encoding (256-dim)
- **Age Embedding**: MLP projection (128-dim)
- **Combined Conditioning**: Concatenated (384-dim total)

---

### 2. **Sub-Components Explained**

#### **CrossAttention Module**
```python
class CrossAttention(nn.Module):
```
- Implements **multi-head self-attention**
- **Purpose**: Captures long-range dependencies in images
- **Parameters**:
  - `channels`: Number of input/output channels
  - `num_heads=8`: Number of attention heads
- **Process**:
  1. Normalize input with GroupNorm
  2. Compute Q, K, V matrices
  3. Calculate attention weights: `Attention = softmax(Q·K^T / √d)`
  4. Apply attention to values: `Output = Attention·V`
  5. Add residual connection

**Where it's used**: Only in bottleneck and last encoder stage to save memory

---

#### **HighQualityResBlock**
```python
class HighQualityResBlock(nn.Module):
```
- **Residual block** with conditional scaling and shifting
- **Parameters**:
  - `in_channels`, `out_channels`: Channel dimensions
  - `cond_dim`: Conditioning vector dimension
  - `use_attention`: Whether to include attention
  
- **Process**:
```
  Input → Conv1 → GroupNorm → SiLU
      ↓
  Add Conditioning (scale & shift)
      ↓
  Conv2 → GroupNorm → [Attention] → SiLU
      ↓
  Add Skip Connection → Output
```

**Conditioning Mechanism**:
```python
scale, shift = condition_projection(cond)
output = output * (1 + scale) + shift
```
This allows the model to adapt features based on the current timestep and plant age.

---

#### **SinusoidalPositionEmbeddings**
```python
class SinusoidalPositionEmbeddings(nn.Module):
```
- Converts **timestep** into a continuous embedding
- **Formula**:
```
  PE(t, 2i)   = sin(t / 10000^(2i/d))
  PE(t, 2i+1) = cos(t / 10000^(2i/d))
```
- **Why?** Helps the model understand which diffusion step it's at
- **Output**: 256-dimensional embedding for each timestep

---

#### **Downsample / Upsample**
```python
class Downsample(nn.Module):  # Reduces spatial resolution by 2x
class Upsample(nn.Module):    # Increases spatial resolution by 2x
```
- **Downsample**: Conv2d with stride=2 (256x256 → 128x128)
- **Upsample**: Bilinear interpolation + Conv2d (128x128 → 256x256)

---

### 3. **PlantDiffusion Class**

Main class managing the diffusion process.

#### **Noise Schedule (Cosine Schedule)**
```python
alpha_bar = cos((t/T + 0.008) / 1.008 * π/2)²
```
- **Purpose**: Controls how much noise is added at each step
- **Cosine schedule** is better than linear for image quality
- Creates smoother transitions between steps

#### **Key Attributes**:
```python
self.betas        # Noise schedule (1000 steps)
self.alphas       # 1 - betas
self.alpha_bars   # Cumulative product of alphas
self.unet         # Main denoising model
self.ema_model    # Exponential Moving Average model (better quality)
```

---

### 4. **DDIM Sampling Algorithm**
```python
@torch.no_grad()
def sample_ddim(self, age, num_samples=1, steps=50, eta=0.0):
```

**DDIM (Denoising Diffusion Implicit Models)** is a faster sampling method:

**Traditional DDPM**: 1000 steps required  
**DDIM**: 50-100 steps (20x faster!)

**Algorithm**:
```
1. Start with random noise: x_T ~ N(0, I)

2. For t = T-1 to 0:
   a. Predict noise: ε_θ(x_t, t, age)
   
   b. Estimate clean image:
      x_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
   
   c. Compute direction to x_t:
      direction = √(1-ᾱ_{t-1}) * ε_θ
   
   d. Update:
      x_{t-1} = √ᾱ_{t-1} * x_0 + direction

3. Return x_0 (generated image)
```

**Parameters**:
- `age`: Plant growth day (0-50)
- `num_samples`: Number of images to generate
- `steps`: Number of denoising steps (higher = better quality, slower)
- `eta`: Stochasticity (0 = deterministic, 1 = random)

---

## Installation

### Requirements
```bash
pip install torch torchvision numpy pillow matplotlib tqdm
```

### Hardware Requirements
- **GPU**: NVIDIA GPU with 6GB+ VRAM (recommended)
- **CPU**: Will work but ~50x slower
- **RAM**: 8GB minimum

---

## Model Files

### Directory Structure
```
project/
├── plant_generator.py              # Main code file
├── best_okra_model.pth             # Okra trained model
├── best_wheat_model.pth            # Wheat trained model
├── best_radish_model.pth           # Radish trained model
├── best_mustard_model.pth          # Mustard trained model
└── README.md                       # This file
```

### Model File Format
Each `.pth` file contains:
```python
{
    'epoch': int,                    # Training epoch number
    'model_state_dict': dict,        # Main UNet weights
    'ema_model_state_dict': dict,    # EMA model weights (better quality)
    'optimizer_state_dict': dict,    # Optimizer state (not used in inference)
    'losses': list                   # Training loss history
}
```

### Updating Model Paths
Edit the `MODEL_PATHS` dictionary:
```python
MODEL_PATHS = {
    'okra': 'path/to/best_okra_model.pth',
    'wheat': 'path/to/best_wheat_model.pth',
    'radish': 'path/to/best_radish_model.pth',
    'mustard': 'path/to/best_mustard_model.pth'
}
```

---

## Usage

### Quick Start
```python
from plant_generator import generate_single_plant_timeline, generate_all_crops_comparison

# Generate okra growth timeline (day 0 to 50, every 5 days)
generate_single_plant_timeline(
    crop_type='okra',
    seed=42,
    start_day=0,
    end_day=50,
    interval=5
)

# Compare all 4 crops at day 25
generate_all_crops_comparison(seed=42, day=25)
```

---

## Code Structure

### 1. **Neural Network Modules** (Lines ~15-300)
```python
CrossAttention              # Multi-head attention mechanism
HighQualityResBlock         # Residual block with conditioning
SinusoidalPositionEmbeddings # Timestep encoding
Downsample / Upsample       # Spatial resolution changes
MustardUNet                 # Main U-Net generator
```

### 2. **Diffusion Management** (Lines ~301-400)
```python
class PlantDiffusion:
    __init__()              # Initialize model and noise schedule
    sample_ddim()           # Generate images using DDIM
    load_checkpoint()       # Load trained model weights
```

### 3. **Generation Functions** (Lines ~401-600)
```python
generate_single_plant_timeline()    # Generate growth timeline for one crop
generate_all_crops_comparison()     # Compare all 4 crops side-by-side
```

### 4. **Configuration** (Lines ~401-410)
```python
MODEL_PATHS = {
    'okra': 'best_okra_model.pth',
    'wheat': 'best_wheat_model.pth',
    'radish': 'best_radish_model.pth',
    'mustard': 'best_mustard_model.pth'
}
```

---

## Functions Reference

### **generate_single_plant_timeline()**

Generate a growth timeline for a single crop.
```python
def generate_single_plant_timeline(
    crop_type="okra",      # Crop to generate ('okra'/'wheat'/'radish'/'mustard')
    seed=42,               # Random seed for reproducibility
    start_day=0,           # Starting day of growth
    end_day=50,            # Ending day of growth
    interval=5,            # Days between each generated image
    output_dir=None        # Output directory (default: '{crop_type}_output')
)
```

**Returns**: Path to the labeled timeline image

**Example Output**:
```
okra_output/
├── okra_seed42_timeline.png          # Raw grid
└── okra_seed42_timeline_labeled.png  # Grid with day labels
```

**Image Layout**:
```
┌─────┬─────┬─────┬─────┬─────┐
│Day 0│Day 5│Day10│Day15│Day20│
├─────┼─────┼─────┼─────┼─────┤
│Day25│Day30│Day35│Day40│Day45│
└─────┴─────┴─────┴─────┴─────┘
```

---

### **generate_all_crops_comparison()**

Compare all 4 crops at the same growth stage.
```python
def generate_all_crops_comparison(
    seed=42,               # Random seed for reproducibility
    day=25,                # Which day to visualize
    output_dir='crops_comparison'  # Output directory
)
```

**Returns**: Path to comparison image

**Example Output**:
```
crops_comparison/
└── all_crops_day25_seed42.png
```

**Image Layout**:
```
┌──────┬───────┬────────┬─────────┐
│ Okra │ Wheat │ Radish │ Mustard │
│Day 25│ Day 25│ Day 25 │ Day 25  │
└──────┴───────┴────────┴─────────┘
```

---

### **PlantDiffusion.sample_ddim()**

Core generation function (usually called internally).
```python
model = PlantDiffusion(img_size=256)
model.load_checkpoint('best_okra_model.pth')

images = model.sample_ddim(
    age=25,              # Day of growth (0-50)
    num_samples=4,       # Number of images to generate
    steps=100,           # Denoising steps (50-100 recommended)
    eta=0.0              # 0=deterministic, 1=stochastic
)
```

**Returns**: Tensor of shape `(num_samples, 3, 256, 256)` in range `[-1, 1]`

**Quality vs Speed**:
- `steps=50`: Fast, good quality
- `steps=100`: Slower, better quality
- `steps=200`: Very slow, marginal improvement

---

## Examples

### Example 1: Generate Timeline for All Crops
```python
from plant_generator import generate_single_plant_timeline

crops = ['okra', 'wheat', 'radish', 'mustard']

for crop in crops:
    print(f"\n=== Generating {crop.upper()} ===")
    generate_single_plant_timeline(
        crop_type=crop,
        seed=42,
        start_day=0,
        end_day=50,
        interval=5
    )
```

**Output**:
```
okra_output/okra_seed42_timeline_labeled.png
wheat_output/wheat_seed42_timeline_labeled.png
radish_output/radish_seed42_timeline_labeled.png
mustard_output/mustard_seed42_timeline_labeled.png
```

---

### Example 2: Compare Growth at Multiple Stages
```python
from plant_generator import generate_all_crops_comparison

# Compare at early, mid, and late growth
for day in [10, 25, 40]:
    print(f"\n=== Day {day} Comparison ===")
    generate_all_crops_comparison(
        seed=42,
        day=day,
        output_dir=f'comparison_day{day}'
    )
```

**Output**:
```
comparison_day10/all_crops_day10_seed42.png
comparison_day25/all_crops_day25_seed42.png
comparison_day40/all_crops_day40_seed42.png
```

---

### Example 3: Generate with Different Seeds
```python
from plant_generator import generate_single_plant_timeline

# Generate 5 different "plants" (same model, different random seeds)
for seed in [10, 20, 30, 40, 50]:
    generate_single_plant_timeline(
        crop_type='okra',
        seed=seed,
        start_day=0,
        end_day=30,
        interval=10,
        output_dir=f'okra_variations'
    )
```

**Output**: 5 different okra growth patterns with natural variation

---

### Example 4: Custom Timeline Range
```python
from plant_generator import generate_single_plant_timeline

# Focus on early growth (first 2 weeks, daily)
generate_single_plant_timeline(
    crop_type='wheat',
    seed=123,
    start_day=0,
    end_day=14,
    interval=1,  # Daily images
    output_dir='wheat_early_growth'
)

# Focus on mature stage (last 10 days, every 2 days)
generate_single_plant_timeline(
    crop_type='wheat',
    seed=123,
    start_day=40,
    end_day=50,
    interval=2,
    output_dir='wheat_mature_stage'
)
```

---

### Example 5: Direct Model Usage (Advanced)
```python
from plant_generator import PlantDiffusion
import torch
from torchvision.utils import save_image

# Load model
model = PlantDiffusion(img_size=256)
model.load_checkpoint('best_mustard_model.pth')

# Generate multiple samples at day 30
torch.manual_seed(42)
images = model.sample_ddim(age=30, num_samples=8, steps=100)

# Post-process
images = (images + 1) / 2  # [-1,1] → [0,1]
images = torch.clamp(images, 0, 1)

# Save
save_image(images, 'mustard_day30_variations.png', nrow=4)
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Error**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size (if generating multiple images):
```python
   # Instead of:
   model.sample_ddim(age=25, num_samples=16)
   
   # Do:
   images = []
   for _ in range(16):
       img = model.sample_ddim(age=25, num_samples=1)
       images.append(img)
```

2. Use CPU (slower):
```python
   device = torch.device('cpu')
```

3. Reduce sampling steps:
```python
   model.sample_ddim(age=25, steps=50)  # Instead of 100
```

---

### Issue 2: Model File Not Found

**Error**:
```
FileNotFoundError: Model not found: best_okra_model.pth
```

**Solutions**:
1. Check file exists:
```python
   import os
   print(os.path.exists('best_okra_model.pth'))
```

2. Use absolute path:
```python
   MODEL_PATHS = {
       'okra': '/full/path/to/best_okra_model.pth',
       ...
   }
```

3. Verify file integrity:
```python
   import torch
   checkpoint = torch.load('best_okra_model.pth')
   print(checkpoint.keys())  # Should show: epoch, model_state_dict, etc.
```

---

### Issue 3: Poor Image Quality

**Problem**: Generated images look noisy or unrealistic

**Solutions**:
1. Increase sampling steps:
```python
   model.sample_ddim(age=25, steps=200)  # Higher quality
```

2. Ensure EMA model is loaded:
```python
   checkpoint = torch.load('best_okra_model.pth')
   print('ema_model_state_dict' in checkpoint)  # Should be True
```

3. Check if model is trained properly:
```python
   checkpoint = torch.load('best_okra_model.pth')
   print(f"Trained for {checkpoint['epoch']} epochs")
   # Should be 100+ epochs
```

---

### Issue 4: Inconsistent Results with Same Seed

**Problem**: Same seed produces different images

**Solution**: Set all random seeds properly:
```python
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# Now generate images
```

---

### Issue 5: Slow Generation Speed

**Problem**: Takes too long to generate images

**Optimization Tips**:

1. Use fewer denoising steps:
```python
   model.sample_ddim(age=25, steps=50)  # Faster
```

2. Enable CUDA if available:
```python
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))
```

3. Batch processing (if memory allows):
```python
   # Generate 10 images in one call
   images = model.sample_ddim(age=25, num_samples=10)
```

4. Use mixed precision (advanced):
```python
   with torch.cuda.amp.autocast():
       images = model.sample_ddim(age=25, num_samples=4)
```

---

## Understanding the Output

### Image Normalization

Generated images go through this pipeline:
```python
# Model outputs in range [-1, 1]
raw_output = model.sample_ddim(age=25)

# Denormalize to [0, 1]
images = (raw_output + 1) / 2

# Clamp to ensure valid range
images = torch.clamp(images, 0, 1)

# Convert to [0, 255] for saving
images_uint8 = (images * 255).byte()
```

### Image Dimensions
```
Tensor Shape: (batch, channels, height, width)
Example: (4, 3, 256, 256)
         ↑  ↑   ↑     ↑
         |  |   |     └── Width (256 pixels)
         |  |   └──────── Height (256 pixels)
         |  └──────────── RGB channels (3)
         └─────────────── Batch size (4 images)
```

---

## Performance Benchmarks

### Generation Time (Single Image, 100 steps)

| Hardware | Time |
|----------|------|
| NVIDIA RTX 4090 | ~3s |
| NVIDIA RTX 3080 | ~5s |
| NVIDIA RTX 2060 | ~10s |
| CPU (Intel i7) | ~3min |

### Memory Usage

| Operation | VRAM |
|-----------|------|
| Model loading | ~2GB |
| Single image (256x256) | ~3GB |
| Batch of 4 images | ~5GB |
| Batch of 8 images | ~8GB |

---

## Advanced Customization

### Modify Image Resolution
```python
# In PlantDiffusion.__init__()
model = PlantDiffusion(img_size=512)  # Higher resolution (needs more VRAM)
```

### Change Number of Diffusion Steps
```python
# In PlantDiffusion.__init__()
model = PlantDiffusion(train_steps=2000)  # More steps = smoother transitions
```

### Adjust Sampling Strategy
```python
# More stochastic (varied outputs)
images = model.sample_ddim(age=25, eta=0.5)

# Completely deterministic
images = model.sample_ddim(age=25, eta=0.0)
```

---

## Citation

If you use this code in your research, please cite:
```bibtex
@software{plant_growth_generator,
  title = {Plant Growth Timeline Generator},
  author = {Your Name},
  year = {2024},
  description = {Diffusion-based plant growth visualization system}
}
```

---

## License

This project is licensed under the MIT License.

---

## Contact & Support

For issues, questions, or contributions:
- GitHub Issues: [your-repo/issues]
- Email: your-email@example.com

---

## Changelog

### Version 1.0.0 (2024)
- ✅ Initial release
- ✅ Support for 4 crop types
- ✅ DDIM sampling implementation
- ✅ Timeline generation
- ✅ Cross-crop comparison

---

**Happy Generating! 🌱🚀**
