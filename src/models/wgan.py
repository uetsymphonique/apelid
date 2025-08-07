import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from utils.logging import get_logger

logger = get_logger(__name__)

class Generator(nn.Module):
    """Generator network for WGAN"""
    def __init__(self, z_dim=100, h_dim=128, x_dim=123):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.x_dim = x_dim
        
        self.model = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()  # Output in [0,1] range
        )
        
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    """Discriminator/Critic network for WGAN"""
    def __init__(self, x_dim=123, h_dim=128):
        super(Discriminator, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        
        self.model = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)  # No sigmoid for WGAN
        )
        
    def forward(self, x):
        return self.model(x)

class DNNCritic(nn.Module):
    """DNN Critic for validating generated samples"""
    def __init__(self, x_dim=123, h_dim=128):
        super(DNNCritic, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        
        self.model = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()  # Output probability
        )
        
    def forward(self, x):
        return self.model(x)

class WGAN:
    """Wasserstein GAN implementation for data augmentation"""
    
    def __init__(self, x_dim=123, z_dim=100, h_dim=128, lr=0.00005, 
                 batch_size=64, weight_clip=0.01, use_gp=False, gp_lambda=10,
                 use_critic_loss=False, lambda_critic=1.0, device='cpu'):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.lr = lr
        self.batch_size = batch_size
        self.weight_clip = weight_clip
        self.use_gp = use_gp
        self.gp_lambda = gp_lambda
        self.use_critic_loss = use_critic_loss
        self.lambda_critic = lambda_critic
        self.device = device
        
        # Initialize networks
        self.generator = Generator(z_dim, h_dim, x_dim).to(device)
        self.discriminator = Discriminator(x_dim, h_dim).to(device)
        self.critic = DNNCritic(x_dim, h_dim).to(device)
        
        # Initialize optimizers
        self.g_optimizer = optim.RMSprop(self.generator.parameters(), lr=lr)
        self.d_optimizer = optim.RMSprop(self.discriminator.parameters(), lr=lr)
        self.c_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        self.c_losses = []
        
        logger.info(f"[+] WGAN initialized with x_dim={x_dim}, z_dim={z_dim}, h_dim={h_dim}")
        if self.use_critic_loss:
            logger.info(f"[+] Using critic feedback with lambda={self.lambda_critic}")
        
    def gradient_penalty(self, real_data, fake_data):
        """Calculate gradient penalty for WGAN-GP"""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand_as(real_data)
        
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated = Variable(interpolated, requires_grad=True)
        
        prob_interpolated = self.discriminator(interpolated)
        
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                      grad_outputs=torch.ones(prob_interpolated.size(), device=self.device),
                                      create_graph=True, retain_graph=True)[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty

    def prepare_data(self, df, use_label_column=False):
        """Prepare data for training"""
        # Separate features and labels
        X = df.drop(columns=['Label']).values.astype(np.float32)
        y = df['Label'].values.astype(np.float32)
        
        # If use_label_column=True we keep original binary labels (0/1),
        # otherwise we default to all-ones (attack) – backward-compatible
        if use_label_column:
            y_binary = y  # assume already 0/1 (Benign=0, Attack=1)
        else:
            # For minority-only training previous logic: all samples are attacks
            y_binary = np.ones_like(y, dtype=np.float32)
        
        # Create tensor dataset
        dataset = Data.TensorDataset(
            torch.tensor(X, device=self.device),
            torch.tensor(y_binary, device=self.device)
        )
        
        # Create data loader
        dataloader = Data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        logger.info(f"[+] Data prepared: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"[+] Labels: {len(y_binary)} samples, all attacks (label=1)")
        return dataloader, X.shape[1]
    
    def train_critic(self, dataloader, epochs=30):
        """Train DNN Critic to distinguish attack vs benign"""
        logger.info(f"[+] Training DNN Critic for {epochs} epochs...")
        
        # Prepare data for critic training
        X_list, y_list = [], []
        for X, y in dataloader:
            X_list.append(X.cpu().numpy())
            y_list.append(y.cpu().numpy())
        
        X_train = np.concatenate(X_list)
        y_train = np.concatenate(y_list)
        
        # Debug: Check data
        logger.info(f"[+] X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
        logger.info(f"[+] y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
        logger.info(f"[+] y_train unique values: {np.unique(y_train)}")
        logger.info(f"[+] y_train min/max: {y_train.min()}/{y_train.max()}")
        
        # Ensure labels are in [0, 1] range for BCELoss
        if y_train.max() > 1 or y_train.min() < 0:
            logger.warning(f"[-] Labels not in [0,1] range, normalizing...")
            y_train = np.clip(y_train, 0, 1)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        
        # Training loop
        for epoch in range(epochs):
            self.critic.train()
            
            # Forward pass
            outputs = self.critic(X_tensor)
            # DNNCritic already has sigmoid, just squeeze
            outputs = outputs.squeeze()
            loss = nn.BCELoss()(outputs, y_tensor)
            
            # Backward pass
            self.c_optimizer.zero_grad()
            loss.backward()
            self.c_optimizer.step()
            
            if epoch % 10 == 0:
                logger.debug(f"[+] Critic Epoch {epoch}, Loss: {loss.item():.4f}")
        
        logger.info(f"[+] DNN Critic training completed")
        return self.critic
    
    def train_wgan(self, dataloader, iterations=50000, d_iter=5, save_interval=1000):
        """Train WGAN"""
        logger.info(f"[+] Starting WGAN training for {iterations} iterations...")
        
        data_iter = iter(dataloader)
        start_time = time.time()
        
        for it in tqdm(range(iterations), desc="Training WGAN"):
            # Train discriminator multiple times
            for _ in range(d_iter):
                try:
                    X_real, _ = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    X_real, _ = next(data_iter)
                
                # Generate fake data
                z = torch.randn(self.batch_size, self.z_dim, device=self.device)
                X_fake = self.generator(z)
                
                # Discriminator forward pass
                D_real = self.discriminator(X_real)
                D_fake = self.discriminator(X_fake.detach())
                
                # WGAN loss
                if self.use_gp:
                    # WGAN-GP
                    gp = self.gradient_penalty(X_real, X_fake.detach())
                    D_loss = -(torch.mean(D_real) - torch.mean(D_fake)) + self.gp_lambda * gp
                else:
                    # Original WGAN
                    D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
                
                # Backward pass
                self.d_optimizer.zero_grad()
                D_loss.backward()
                self.d_optimizer.step()
                
                # Weight clipping (only for original WGAN)
                if not self.use_gp:
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-self.weight_clip, self.weight_clip)
            
            # Train generator
            z = torch.randn(self.batch_size, self.z_dim, device=self.device)
            X_fake = self.generator(z)
            D_fake = self.discriminator(X_fake)
            G_loss = -torch.mean(D_fake)
            if self.use_critic_loss:
                # Freeze critic params
                self.critic.eval()
                for p in self.critic.parameters():
                    p.requires_grad = False
                critic_out = self.critic(X_fake)
                # we want critic_out -> 1, so loss = -log(score)
                val_loss = -torch.mean(torch.log(critic_out + 1e-8))
                G_loss = G_loss + self.lambda_critic * val_loss
            
            self.g_optimizer.zero_grad()
            G_loss.backward()
            self.g_optimizer.step()
            
            # Record losses
            self.g_losses.append(G_loss.item())
            self.d_losses.append(D_loss.item())
            
            # Logging
            if it % save_interval == 0:
                elapsed = time.time() - start_time
                logger.info(f"[+] Iter {it}: G_loss={G_loss.item():.4f}, D_loss={D_loss.item():.4f}, Time={elapsed:.1f}s")
        
        logger.info(f"[+] WGAN training completed")
    
    def generate_samples(self, num_samples, critic_threshold=0.7):
        """Generate samples using trained generator and validate with critic"""
        logger.info(f"[+] Generating {num_samples} samples...")
        
        self.generator.eval()
        self.critic.eval()
        
        generated_samples = []
        valid_samples = 0
        total_attempts = 0
        
        with torch.no_grad():
            while len(generated_samples) < num_samples and total_attempts < num_samples * 10:  # Safety limit
                # Generate batch
                batch_size = min(self.batch_size, num_samples - len(generated_samples))
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                fake_data = self.generator(z)
                
                # Validate with critic
                critic_scores = self.critic(fake_data)
                # ---- verbose logging ----
                if total_attempts % (self.batch_size * 5) == 0:  # log mỗi 5 batch
                    logger.debug(
                        f"Batch stats | min: {critic_scores.min().item():.3f} "
                        f"max: {critic_scores.max().item():.3f} "
                        f"mean: {critic_scores.mean().item():.3f}")
                valid_mask = (critic_scores.squeeze() > critic_threshold)
                
                # Keep valid samples
                valid_fake = fake_data[valid_mask]
                if valid_fake.size(0) > 0:
                    logger.debug(f"Valid ratio: {valid_fake.size(0)}/{batch_size} "
                                 f"= {valid_fake.size(0)/batch_size:.1%}")
                if len(valid_fake) > 0:
                    # Convert to numpy and ensure 2D shape
                    valid_numpy = valid_fake.cpu().numpy()
                    if valid_numpy.ndim == 1:
                        valid_numpy = valid_numpy.reshape(1, -1)
                    generated_samples.append(valid_numpy)
                    valid_samples += len(valid_fake)
                
                total_attempts += batch_size
                logger.debug(f"[+] Generated batch: {len(valid_fake)}/{batch_size} valid samples")
        
        if not generated_samples:
            logger.warning("[-] No valid samples generated, returning random samples")
            # Fallback: generate random samples
            z = torch.randn(num_samples, self.z_dim, device=self.device)
            fake_data = self.generator(z)
            return fake_data.cpu().numpy()
        
        # Combine all samples
        try:
            all_samples = np.concatenate(generated_samples, axis=0)
            logger.info(f"[+] Generated {len(all_samples)} valid samples out of {total_attempts} attempts")
            
            # Ensure we have exactly num_samples
            if len(all_samples) > num_samples:
                all_samples = all_samples[:num_samples]
            elif len(all_samples) < num_samples:
                logger.warning(f"[-] Only generated {len(all_samples)} samples, requested {num_samples}")
            
            return all_samples
            
        except ValueError as e:
            logger.error(f"[-] Error concatenating samples: {e}")
            logger.error(f"[-] Sample shapes: {[s.shape for s in generated_samples]}")
            # Fallback: return first batch
            return generated_samples[0]
    
    def save_models(self, save_dir="models"):
        """Save trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save WGAN models
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses
        }, os.path.join(save_dir, 'wgan_model.pth'))
        
        # Save critic model
        torch.save({
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer_state_dict': self.c_optimizer.state_dict(),
            'critic_losses': self.c_losses
        }, os.path.join(save_dir, 'critic_model.pth'))
        
        logger.info(f"[+] Models saved to {save_dir}")
    
    def load_models(self, save_dir="models"):
        """Load trained models"""
        # Load WGAN models
        wgan_checkpoint = torch.load(os.path.join(save_dir, 'wgan_model.pth'), map_location=self.device)
        self.generator.load_state_dict(wgan_checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(wgan_checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(wgan_checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(wgan_checkpoint['d_optimizer_state_dict'])
        self.g_losses = wgan_checkpoint['g_losses']
        self.d_losses = wgan_checkpoint['d_losses']
        
        # Load critic model
        critic_checkpoint = torch.load(os.path.join(save_dir, 'critic_model.pth'), map_location=self.device)
        self.critic.load_state_dict(critic_checkpoint['critic_state_dict'])
        self.c_optimizer.load_state_dict(critic_checkpoint['critic_optimizer_state_dict'])
        self.c_losses = critic_checkpoint['critic_losses']
        
        logger.info(f"[+] Models loaded from {save_dir}")
    
    def plot_losses(self, save_path="wgan_losses.png"):
        """Plot training losses"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.g_losses, label='Generator Loss', alpha=0.7)
        plt.plot(self.d_losses, label='Discriminator Loss', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('WGAN Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if self.c_losses:
            plt.subplot(1, 2, 2)
            plt.plot(self.c_losses, label='Critic Loss', color='green', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('DNN Critic Training Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[+] Loss plot saved to {save_path}") 