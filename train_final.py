import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import os
import yaml
from dataclasses import dataclass, asdict
# --- 1. Training Configuration ---
@dataclass
class Config:
    latent_dim: int = 16
    data_dim: int = 2
    N: int = 512
    T: float = 0.1            # Kernel bandwidth (Temperature)
    epochs: int = 100010      # Steps actually
    lr: float = 1e-3
    gamma_start: float = 0.2  # Initial mode-seeking strength
    gamma_end: float = 1.5    # Final mode-seeking strength
    warmup: int = 15000       # Linear warmup steps for gamma
    plot_interval: int = 500
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir: str = "training_plots"
    # A dirty hack to allow saving/loading device info in YAML
    def save_to_yaml(self, path):
        """Saves current configuration to a YAML file."""
        with open(path, 'w') as f:
            # We convert the dataclass to a dict, then to yaml
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def load_from_yaml(cls, path):
        """Loads configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

# --- 2. Model Architecture ---
class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, data_dim)
        )

    def forward(self, e):
        return self.net(e)

# --- 3. Functional Logic & Formulas ---

def sample_real_data(n, device):
    """Generates an 8-mode Gaussian mixture ring."""
    indices = torch.randint(0, 8, (n,), device=device)
    angles = indices.float() * (2 * np.pi / 8)
    centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 3.0
    return centers + torch.randn(n, 2, device=device) * 0.2

def compute_V_DMD_Strict(x, y_pos, y_neg, T, gamma):
    """
    Implements the score-matching drift field.
    Formula: V = gamma * (Barycenter_pos - x) - (Barycenter_neg - x)
    This ensures p_fake matches p_real^gamma at equilibrium (V=0).
    """
    # Positive Score Proxy (Attraction to real data)
    # This is the mean-shift vector pointing towards the weighted barycenter of y_pos
    # which is a proxy for the (smoothed) score of p_real(x) ^ gamma
    dist_pos = torch.cdist(x, y_pos)
    w_pos = F.softmax(-dist_pos**2 / T, dim=1) 
    v_pos = (w_pos @ y_pos) - x
    
    # Negative Score Proxy (Repulsion from generated batch)
    dist_neg = torch.cdist(x, y_neg)
    # Mask self-comparison to prevent infinite attraction to self
    dist_neg = dist_neg + torch.eye(x.size(0), device=x.device) * 1e6
    w_neg = F.softmax(-dist_neg**2 / T, dim=1)
    v_neg = (w_neg @ y_neg) - x
    
    return gamma * v_pos - v_neg

def get_density_data(gamma, device, grid_size=80):
    """Calculates density grids for the original and gamma-scaled distributions."""
    indices = torch.arange(8, device=device)
    angles = indices.float() * (2 * np.pi / 8)
    centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 3.0
    
    x_range = torch.linspace(-5, 5, grid_size, device=device)
    y_range = torch.linspace(-5, 5, grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')
    points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    
    dists = torch.cdist(points, centers)
    p_x = torch.exp(-0.5 * (dists / 0.3)**2).sum(dim=1)
    
    p_real = p_x.reshape(grid_size, grid_size).cpu().numpy()
    p_gamma = torch.pow(p_x, gamma).reshape(grid_size, grid_size).cpu().numpy()
    
    return grid_x.cpu().numpy(), grid_y.cpu().numpy(), p_real, p_gamma

# --- 4. Plotting Module ---

def update_plots(axs, fig, epoch, x, V, loss_history, gamma, cfg):
    """Handles the 4-panel dashboard updates."""
    if not plt.fignum_exists(fig.number):
        return False

    gx, gy, gz_real, gz_gamma = get_density_data(gamma, cfg.device)
    grid_lim = [-5, 5]
    common_cmap = 'YlGnBu'
    sample_color = 'orangered'
    bg_color = '#f8f8f8'

    # Panel 1: Original p_real
    axs[0,0].clear()
    axs[0,0].contourf(gx, gy, gz_real, levels=15, cmap=common_cmap, alpha=0.8)
    axs[0,0].set_title("Original $p_{real}$ Reference")
    
    # Panel 2: Target p_real^gamma + Samples
    axs[0,1].clear()
    axs[0,1].contourf(gx, gy, gz_gamma, levels=15, cmap=common_cmap, alpha=0.8)
    axs[0,1].scatter(x.detach().cpu()[:,0], x.detach().cpu()[:,1], 
                     c=sample_color, s=7, edgecolors='white', linewidths=0.2)
    axs[0,1].set_title(f"Target $p_{{real}}^{{{gamma:.2f}}}$ + Samples")

    # Panel 3: Vector Field V
    axs[1,0].clear()
    v_c, x_c = V.detach().cpu().numpy(), x.detach().cpu().numpy()
    axs[1,0].quiver(x_c[:, 0], x_c[:, 1], v_c[:, 0], v_c[:, 1], color='forestgreen', alpha=0.5, scale=20)
    axs[1,0].set_title("Drift Field $V$ (Score Matching)")

    # Panel 4: Loss History
    axs[1,1].clear()
    axs[1,1].plot(loss_history, color='indigo', linewidth=1.5)
    axs[1,1].set_yscale('log')
    axs[1,1].set_title("Training Loss: $\|V\|^2$")

    for i, j in [(0,0), (0,1), (1,0)]:
        axs[i,j].set_xlim(grid_lim); axs[i,j].set_ylim(grid_lim)
        axs[i,j].set_aspect('equal'); axs[i,j].set_facecolor(bg_color)
    
    file_name = f"epoch_{epoch:06d}.png"
    # make sure save directory exists
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, file_name)
    fig.savefig(save_path, dpi=120)

    plt.tight_layout()
    plt.pause(0.01)
    return True

# --- 5. Main Training Function ---

def train(cfg):
    model = Generator(cfg.latent_dim, cfg.data_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_history = []

    # UI Setup
    plt.ion()
    fig, axs = plt.subplots(2, 2, figsize=(13, 11))
    
    is_running = [True]
    def on_close(event):
        is_running[0] = False
        print("Training interrupted by user.")
    fig.canvas.mpl_connect('close_event', on_close)

    print(f"Starting training for {cfg.epochs} epochs...")

    for epoch in range(cfg.epochs):
        if not is_running[0]: break

        # Gamma Linear Schedule
        curr_gamma = cfg.gamma_start + (cfg.gamma_end - cfg.gamma_start) * min(1.0, epoch / cfg.warmup)

        # 1. Generate Samples
        e = torch.randn(cfg.N, cfg.latent_dim, device=cfg.device)
        x = model(e)
        
        # 2. Compute Target via Score-Matching Drift
        y_pos = sample_real_data(cfg.N, cfg.device)
        with torch.no_grad():
            V = compute_V_DMD_Strict(x, y_pos, x.clone(), cfg.T, curr_gamma)
            x_target = x + V
        
        # 3. Optimization Step
        loss = F.mse_loss(x, x_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())

        # 4. Visualization
        if epoch % cfg.plot_interval == 0:
            if not update_plots(axs, fig, epoch, x, V, loss_history, curr_gamma, cfg):
                break

    plt.ioff()
    plt.show()

import sys

if __name__ == "__main__":
    # Check if a yaml path was passed as a command line argument
    # Usage: python script.py my_config.yaml
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        cfg = Config.load_from_yaml(config_path)
        print(f"Loaded config from {config_path}")
    else:
        cfg = Config()
        print("Using default configuration")

    # Ensure device is converted to torch object if it was loaded as a string
    cfg.device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
    # Print the configuration for verification
    print("Training Configuration:")
    for field in cfg.__dataclass_fields__:
        print(f"  {field}: {getattr(cfg, field)}")
    train(cfg)