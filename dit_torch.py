import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Hyperparameters
T = 1.0  # Flow matching time horizon
d_model = 128  # Reduced for debugging
num_layers = 8  # Reduced for debugging
num_heads = 8  # Adjusted for d_model
ff_dim = 512  # Reduced for debugging
patch_size = 4
num_patches = (28 // patch_size) ** 2
batch_size = 16
max_steps = 100000  # Reduced for faster debugging
sample_interval = 500  # Save samples more frequently
log_interval = 100
num_samples = 16
output_dir = "overfit_samples_debug"
log_file = "overfit_log_debug.txt"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Sinusoidal time embeddings
def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :].to(timesteps.device)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb

# Patch embedding module
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * patch_size, d_model)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.reshape(B, -1, self.patch_size * self.patch_size)
        x = self.proj(x)
        return x

# Flow Matching Transformer model
class FlowMatchingTransformer(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, ff_dim, num_patches):
        super().__init__()
        self.patch_emb = PatchEmbedding(patch_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        self.time_emb = nn.Linear(d_model, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout=0.1, batch_first=True),
            num_layers
        )
        self.out_proj = nn.Linear(d_model, patch_size * patch_size)
    
    def forward(self, x, t):
        B = x.shape[0]
        x = self.patch_emb(x)  # (B, num_patches, d_model)
        x = x + self.pos_enc
        t_emb = get_timestep_embedding(t, d_model).to(x.device)
        t_emb = self.time_emb(t_emb)  # (B, d_model)
        t_emb = t_emb[:, None, :]  # (B, 1, d_model)
        x = x + t_emb
        x = self.transformer(x)  # (B, num_patches, d_model)
        x = self.out_proj(x)  # (B, num_patches, patch_size * patch_size)
        x = x.view(B, 1, 28, 28)
        return x

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlowMatchingTransformer(d_model, num_layers, num_heads, ff_dim, num_patches).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # Fallback learning rate
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

# Load MNIST dataset and select one batch
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
data_iterator = iter(train_loader)
x1, _ = next(data_iterator)
x1 = x1.to(device)

# Verify data
print(f"x1 shape: {x1.shape}, min: {x1.min().item():.4f}, max: {x1.max().item():.4f}")

# Save original batch
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(x1[i, 0].cpu(), cmap='gray')
    ax.axis('off')
plt.suptitle('Original Batch')
plt.savefig(os.path.join(output_dir, 'original_batch.png'), bbox_inches='tight')
plt.close()

# Sampling function
def sample(model, num_samples, steps=100):
    model.eval()
    with torch.no_grad():
        x = torch.randn(num_samples, 1, 28, 28, device=device)
        dt = T / steps
        for i in range(steps):
            t = torch.full((num_samples,), i * dt, device=device)
            v = model(x, t)
            x = x + v * dt
        x = torch.clamp(x, 0, 1)
    return x

# Save generated images
def save_samples(images, step, output_dir):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0].cpu(), cmap='gray')
        ax.axis('off')
    plt.suptitle(f'Step {step}')
    plt.savefig(os.path.join(output_dir, f'samples_step_{step}.png'), bbox_inches='tight')
    plt.close()

# Training loop with debugging
with open(log_file, 'w') as f:
    f.write('Step,Loss,Per_Sample_Loss,Grad_Norm,Learning_Rate\n')
for step in range(max_steps):
    model.train()
    t = torch.rand(batch_size, device=device)
    x0 = torch.randn_like(x1)
    xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
    target_v = x1 - x0
    pred_v = model(xt, t)
    
    # Verify shapes and values
    if step == 0:
        print(f"xt shape: {xt.shape}, min: {xt.min().item():.4f}, max: {xt.max().item():.4f}")
        print(f"target_v shape: {target_v.shape}, mean: {target_v.mean().item():.4f}, std: {target_v.std().item():.4f}")
        print(f"pred_v shape: {pred_v.shape}, mean: {pred_v.mean().item():.4f}, std: {pred_v.std().item():.4f}")
    
    # Compute per-sample loss
    per_sample_loss = F.mse_loss(pred_v, target_v, reduction='none').mean(dim=(1, 2, 3))
    loss = per_sample_loss.mean()
    
    optimizer.zero_grad()
    loss.backward()
    
    # Compute gradient norm
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()
    
    # Log
    if (step + 1) % log_interval == 0 or step == 0:
        current_lr = scheduler.get_last_lr()[0]
        per_sample_loss_str = ','.join([f'{l:.4f}' for l in per_sample_loss.detach().cpu().numpy()])
        print(f'Step {step+1}, Loss: {loss:.4f}, Grad Norm: {grad_norm:.4f}, LR: {current_lr:.6f}')
        print(f'Per-sample Loss: [{per_sample_loss_str}]')
        with open(log_file, 'a') as f:
            f.write(f'{step+1},{loss:.4f},"[{per_sample_loss_str}]",{grad_norm:.4f},{current_lr:.6f}\n')
    
    # Save samples
    if (step + 1) % sample_interval == 0:
        generated_images = sample(model, num_samples)
        save_samples(generated_images, step + 1, output_dir)
        print(f'Saved {num_samples} images for step {step+1} to {output_dir}')
