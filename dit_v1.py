import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax.random import PRNGKey
from tensorflow.keras.datasets import mnist
import numpy as np
from typing import Any, Tuple
from functools import partial

# Constants
IMG_SIZE = 28
PATCH_SIZE = 4
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 12
BATCH_SIZE = 64
NUM_CLASSES = 10
EPOCHS = 10
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.1
T_MIN = 1e-4  # Minimum time for flow matching
T_MAX = 1.0    # Maximum time for flow matching

# Data preprocessing
def load_mnist():
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_train = x_train[:, :, :, None]  # Add channel dimension
    y_train = jax.nn.one_hot(y_train, NUM_CLASSES)
    return x_train, y_train

# Flow matching utilities
def flow_matching_path(x0, x1, t):
    """Linear interpolation path: x_t = (1-t)x0 + t*x1"""
    return (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1

def flow_matching_velocity(x0, x1, t):
    """Velocity field: v_t = x1 - x0"""
    return x1 - x0

# Adaptive Layer Normalization
class AdaLN(nn.Module):
    embed_dim: int
    
    @nn.compact
    def __call__(self, x, t_emb):
        scale = nn.Dense(self.embed_dim)(t_emb)
        shift = nn.Dense(self.embed_dim)(t_emb)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]

# Transformer Block
class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    dropout_rate: float
    
    @nn.compact
    def __call__(self, x, t_emb, training: bool):
        # Multi-head self-attention
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            dropout_rate=self.dropout_rate if training else 0.0
        )(x, deterministic=not training)
        
        # Residual connection and normalization
        x = nn.LayerNorm()(x + attn)
        
        # AdaLN modulation
        x = AdaLN(self.embed_dim)(x, t_emb)
        
        # Feed-forward network
        ffn = nn.Sequential([
            nn.Dense(self.embed_dim * 4),
            nn.gelu,
            nn.Dense(self.embed_dim),
            nn.Dropout(self.dropout_rate, deterministic=not training)
        ])(x)
        
        # Residual connection and normalization
        x = nn.LayerNorm()(x + ffn)
        return x

# DiT Model
class DiT(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    patch_size: int
    num_classes: int
    dropout_rate: float
    
    def setup(self):
        self.patch_embed = nn.Dense(self.embed_dim)
        num_patches = (IMG_SIZE // self.patch_size) ** 2
        self.pos_embed = self.param('pos_embed', 
                                  nn.initializers.normal(0.02),
                                  (1, num_patches, self.embed_dim))
        self.class_embed = nn.Dense(self.embed_dim)
        self.time_embed = nn.Sequential([
            nn.Dense(self.embed_dim),
            nn.gelu,
            nn.Dense(self.embed_dim)
        ])
        self.blocks = [TransformerBlock(self.embed_dim, self.num_heads, self.dropout_rate) 
                      for _ in range(self.num_layers)]
        self.output_layer = nn.Dense(self.patch_size * self.patch_size)
    
    def __call__(self, x, t, class_labels, training=False):
        # Patchify input
        patches = x.reshape(x.shape[0], -1, self.patch_size * self.patch_size)
        x = self.patch_embed(patches) + self.pos_embed
        
        # Time and class embeddings
        t = t[:, None]
        t_emb = self.time_embed(t)
        c_emb = self.class_embed(class_labels)
        t_emb = t_emb + c_emb  # Combine time and class embeddings
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, training)
        
        # Output: predict velocity
        x = self.output_layer(x)
        x = x.reshape(x.shape[0], IMG_SIZE, IMG_SIZE, 1)
        return x

# Training step
@partial(jax.jit, static_argnums=(0,))
def train_step(model, params, opt_state, x, y, rng):
    def loss_fn(params):
        rng, noise_rng = jax.random.split(rng)
        t = jax.random.uniform(noise_rng, (x.shape[0],), minval=T_MIN, maxval=T_MAX)
        noise = jax.random.normal(noise_rng, x.shape)
        x1 = x  # Data
        x0 = noise  # Noise
        x_t = flow_matching_path(x0, x1, t)
        v_target = flow_matching_velocity(x0, x1, t)
        v_pred = model.apply({'params': params}, x_t, t, y, training=True, 
                            rngs={'dropout': noise_rng})
        loss = jnp.mean((v_pred - v_target) ** 2)
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Sampling function
def sample(model, params, rng, num_samples, class_labels):
    x = jax.random.normal(rng, (num_samples, IMG_SIZE, IMG_SIZE, 1))
    dt = (T_MAX - T_MIN) / TIMESTEPS
    for t in np.linspace(T_MAX, T_MIN, TIMESTEPS):
        t_array = jnp.full((num_samples,), t, dtype=jnp.float32)
        v_pred = model.apply({'params': params}, x, t_array, class_labels, training=False)
        x = x + v_pred * dt
    return jnp.clip(x, 0.0, 1.0)

# Main
if __name__ == "__main__":
    rng = PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    # Load and preprocess data
    x_train, y_train = load_mnist()
    num_batches = len(x_train) // BATCH_SIZE
    
    # Initialize model and optimizer
    model = DiT(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                patch_size=PATCH_SIZE, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE)
    dummy_x = jnp.ones((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1))
    dummy_t = jnp.ones((BATCH_SIZE,))
    dummy_y = jnp.ones((BATCH_SIZE, NUM_CLASSES))
    params = model.init({'params': init_rng, 'dropout': init_rng}, dummy_x, dummy_t, dummy_y, training=True)['params']
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)
    
    # Training loop
    for epoch in range(EPOCHS):
        rng, epoch_rng = jax.random.split(rng)
        indices = jax.random.permutation(epoch_rng, len(x_train))
        total_loss = 0.0
        for i in range(num_batches):
            batch_indices = indices[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]
            rng, batch_rng = jax.random.split(rng)
            params, opt_state, loss = train_step(model, params, opt_state, x_batch, y_batch, batch_rng)
            total_loss += loss
        print(f"Epoch {epoch + 1}, Loss: {total_loss / num_batches:.4f}")
    
    # Generate samples
    rng, sample_rng = jax.random.split(rng)
    sample_classes = jax.nn.one_hot(jnp.arange(NUM_CLASSES), NUM_CLASSES)
    samples = sample(model, params, sample_rng, NUM_CLASSES, sample_classes)
    print("Generated samples shape:", samples.shape)
