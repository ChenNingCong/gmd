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
    seed: int = 42
    latent_dim: int = 16
    data_dim: int = 2
    N: int = 512
    N_plot: int = 4096
    # By default, we keep temperature constant, but these parameters allow for linear decay if desired
    T_start: float = 0.1      # Initial temperature for kernel
    T_end: float = 0.1       # Final temperature for kernel
    T_warmup_epochs: int = 15000       # Linear decay steps for temperature
    epochs: int = 100010      # Steps actually
    lr: float = 1e-3
    gamma_start: float = 0.2  # Initial mode-seeking strength
    gamma_end: float = 1.5    # Final mode-seeking strength
    gamma_warmup_epochs: int = 15000       # Linear warmup steps for gamma
    plot_interval: int = 1000
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir: str = "training_plots"
    # This can be kl_divergence, sinkhorn, sinkhorn_col or sinkhorn_col_row
    mmd_method: str = "kl_divergence"
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
        # replace each layer with a residual block to help with training stability

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, data_dim)
        )
        # # zero output layer initialization to start with samples near the origin
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, e):
        return self.net(e)

# --- 3. Functional Logic & Formulas ---

# def sample_real_data(n, device):
#     """Generates an 8-mode Gaussian mixture ring."""
#     indices = torch.randint(0, 8, (n,), device=device)
#     angles = indices.float() * (2 * np.pi / 8)
#     centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 3.0
#     return centers + torch.randn(n, 2, device=device) * 0.2

# def get_density_data(gamma, device, grid_size=80):
#     """Calculates density grids for the original and gamma-scaled distributions."""
#     indices = torch.arange(8, device=device)
#     angles = indices.float() * (2 * np.pi / 8)
#     centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 3.0
    
#     x_range = torch.linspace(-5, 5, grid_size, device=device)
#     y_range = torch.linspace(-5, 5, grid_size, device=device)
#     grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')
#     points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    
#     dists = torch.cdist(points, centers)
#     p_x = torch.exp(-0.5 * (dists / 0.3)**2).sum(dim=1)
    
#     p_real = p_x.reshape(grid_size, grid_size).cpu().numpy()
#     p_gamma = torch.pow(p_x, gamma).reshape(grid_size, grid_size).cpu().numpy()
    
#     return grid_x.cpu().numpy(), grid_y.cpu().numpy(), p_real, p_gamma
# def sample_real_data(n, device):
#     """Blazing fast checkerboard sampling using direct coordinate mapping."""
#     # 1. Randomly pick which 'black' squares to occupy (out of 32 possible)
#     # Total squares = 64, Black squares = 32
#     cell_indices = torch.randint(0, 32, (n,), device=device)
    
#     # 2. Map 1D index (0-31) to 2D checkerboard coordinates (0-7, 0-7)
#     row = torch.div(cell_indices, 4, rounding_mode='floor')
#     col = (cell_indices % 4) * 2 + (row % 2)
    
#     # 3. Scale to actual coordinate range (e.g., -4 to 4)
#     # Shift indices to centered coordinates and add uniform noise [-0.5, 0.5]
#     x = (col.float() - 3.5) + (torch.rand(n, device=device) - 0.5)
#     y = (row.float() - 3.5) + (torch.rand(n, device=device) - 0.5)
    
#     return torch.stack([x, y], dim=1)

# def get_density_data(gamma, device, grid_size=80):
#     """Vectorized density calculation."""
#     x_range = torch.linspace(-4, 4, grid_size, device=device)
#     y_range = torch.linspace(-4, 4, grid_size, device=device)
#     grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')
    
#     # Vectorized parity check: (floor(x+4) + floor(y+4)) % 2
#     # We use a small epsilon to avoid boundary artifacts
#     p_x = ((torch.floor(grid_x + 4.0).int() + torch.floor(grid_y + 4.0).int()) % 2 == 0).float()
    
#     # Add a tiny epsilon before gamma so 0^gamma doesn't explode if gamma is negative
#     p_real = p_x.cpu().numpy()
#     p_gamma = torch.pow(p_x + 1e-9, gamma).cpu().numpy()
    
#     return grid_x.cpu().numpy(), grid_y.cpu().numpy(), p_real, p_gamma

import torch
import numpy as np

def sample_real_data(n, device):
    """
    生成高密度的瑞士卷分布（约 4.5 圈）。
    """
    # 1. 增大 t 的范围。t 的跨度越大，圈数越多。
    # 范围从 1.5π 到 10.5π (10.5 - 1.5 = 9π，即 4.5 圈)
    t = (1.5 * np.pi) + (9.0 * np.pi * torch.rand(n, device=device))
    
    # 2. 参数化方程
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    
    # 3. 添加轻微噪声，保持流形厚度
    # 随着圈数增加，噪声比例需要调小，否则外圈会糊在一起
    noise = 0.1
    x += noise * torch.randn(n, device=device)
    y += noise * torch.randn(n, device=device)
    
    # 4. 缩放系数：因为 t 变大了（最大约 33），我们需要缩小系数 (约 0.15)
    # 这样坐标依然保持在 [-5, 5] 左右
    return torch.stack([x, y], dim=1) * 0.15

def get_density_data(gamma, device, grid_size=120):
    """
    计算高密度瑞士卷的概率密度地图。
    """
    # 1. 设置网格
    limit = 6.0
    x_range = torch.linspace(-limit, limit, grid_size, device=device)
    y_range = torch.linspace(-limit, limit, grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')
    grid_points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    
    # 2. 生成极高分辨率的参考流形（为了让密度图更丝滑）
    ref_t = torch.linspace(1.5 * np.pi, 10.5 * np.pi, 5000, device=device)
    ref_x = (ref_t * torch.cos(ref_t) * 0.15)
    ref_y = (ref_t * torch.sin(ref_t) * 0.15)
    ref_points = torch.stack([ref_x, ref_y], dim=1)

    # 3. 计算每个网格点到螺旋线的最小距离
    dist = torch.cdist(grid_points, ref_points, p=2)
    min_dist, _ = torch.min(dist, dim=1)
    
    # 4. 映射为密度分布
    # 圈数多了，sigma 要调小一点，否则层与层之间会连起来
    sigma = 0.12
    density = torch.exp(-0.5 * (min_dist / sigma)**2)
    
    p_real = density.reshape(grid_size, grid_size).cpu().numpy()
    p_gamma = torch.pow(torch.tensor(p_real) + 1e-9, gamma).numpy()
    
    return grid_x.cpu().numpy(), grid_y.cpu().numpy(), p_real, p_gamma
def compute_V_DMD_kl_divergence(x, y_pos, y_neg, T, gamma):
    """
    Implements the score-matching drift field.
    Formula: V = gamma * (Barycenter_pos - x) - (Barycenter_neg - x)
    This ensures p_fake matches p_real^gamma at equilibrium (V=0).
    """
    # Positive Score Proxy (Attraction to real data)
    # This is the mean-shift vector pointing towards the weighted barycenter of y_pos
    # which is a proxy for the (smoothed) score of p_real(x) ^ gamma
    dist_pos = torch.cdist(x, y_pos)
    w_pos = F.softmax(-dist_pos / T, dim=1) 
    v_pos = (w_pos @ y_pos) - x
    
    # Negative Score Proxy (Repulsion from generated batch)
    dist_neg = torch.cdist(x, y_neg)
    # Mask self-comparison to prevent infinite attraction to self
    dist_neg = dist_neg + torch.eye(x.size(0), device=x.device) * 1e6
    w_neg = F.softmax(-dist_neg / T, dim=1)
    v_neg = (w_neg @ y_neg) - x
    
    return gamma * v_pos - v_neg

def compute_V_DMD_Strict_sinkhorn_col_row(x, y_pos, y_neg, T=0.1, gamma=0.0):
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    
    dist_pos = torch.cdist(x, y_pos) 
    dist_neg = torch.cdist(x, y_neg)
    dist_neg = dist_neg + torch.eye(N, device=x.device) * 1e6

    # When gamma is 0, logit_pos becomes 0 (uniform weight)
    # This removes the directional preference toward real data.
    logit_pos = (-dist_pos / T)
    logit_neg = (-dist_neg / T) # Keep repulsion active
    
    logit = torch.cat([logit_pos, logit_neg], dim=1)
    A_row = F.softmax(logit, dim=-1)
    A_col = F.softmax(logit, dim=-2)
    A = torch.sqrt(A_row * A_col)

    A_pos, A_neg = torch.split(A, [N_pos, N], dim=1)
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

    V = (gamma * (W_pos @ y_pos - x)) - (W_neg @ y_neg - x)
    return V


def compute_V_DMD_Strict_sinkhorn_col(x, y_pos, y_neg, T=0.1, gamma=0.0):
    # only normalize x
    assert gamma == 1.0, "This function is designed for gamma=1.0 (original DMD). For other gamma values, use compute_V_DMD_Strict."
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    
    dist_pos = torch.cdist(x, y_pos) 
    dist_neg = torch.cdist(x, y_neg)
    dist_neg = dist_neg + torch.eye(N, device=x.device) * 1e6

    # When gamma is 0, logit_pos becomes 0 (uniform weight)
    # This removes the directional preference toward real data.
    logit_pos = (-dist_pos / T)
    logit_neg = (-dist_neg / T) # Keep repulsion active

    A_pos = F.softmax(logit_pos, dim=0)
    A_neg = F.softmax(logit_neg, dim=0)
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

    V = W_pos @ y_pos - W_neg @ y_neg
    return V


def compute_V_DMD_Strict_sinkhorn(x, y_pos, y_neg, eps=0.1, n_iters=5, gamma=1.0):
    """
    Implements the drift field using Sinkhorn Divergence (Optimal Transport).
    
    Args:
        x: Current particles/samples [Batch, Dim]
        y_pos: Target 'real' samples [Batch, Dim]
        y_neg: Negative 'fake' samples [Batch, Dim]
        eps: Regularization strength (smaller = closer to pure Wasserstein)
        n_iters: Number of Sinkhorn iterations
        gamma: Weighting factor for the positive drift
    """
    
    def get_sinkhorn_drift(source, target):
        # 1. Compute Cost Matrix (Squared L2 Distance)
        C = torch.cdist(source, target) ** 2
        
        # 2. Sinkhorn Iterations (Matrix Scaling)
        # We find the transport plan P that is doubly stochastic
        K = torch.exp(-C / eps)
        u = torch.ones(source.size(0), device=source.device) / source.size(0)
        v = torch.ones(target.size(0), device=target.device) / target.size(0)
        
        for _ in range(n_iters):
            u = 1.0 / (K @ v + 1e-8)
            v = 1.0 / (K.T @ u + 1e-8)
            
        # 3. Optimal Transport Plan P
        P = u.unsqueeze(1) * K * v.unsqueeze(0)
        
        # 4. Compute the barycentric projection
        # This scales P so each row sums to 1 to project 'source' onto 'target'
        row_sums = P.sum(dim=1, keepdim=True)
        P_normalized = P / (row_sums + 1e-8)
        
        # Drift vector: (Target_Barycenter - Source)
        return (P_normalized @ target) - source

    # Positive Drift (Matching the real distribution)
    v_pos = get_sinkhorn_drift(x, y_pos)
    
    # Negative Drift (Avoiding the fake distribution)
    v_neg = get_sinkhorn_drift(x, y_neg)
    
    return gamma * v_pos - v_neg

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
    # Overlay real data samples for reference
    y_ref = sample_real_data(cfg.N_plot, cfg.device).cpu().numpy()
    axs[0,0].scatter(y_ref[:,0], y_ref[:,1], c=sample_color, s=7, edgecolors='white', linewidths=0.2)
    
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

def train(cfg : Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = Generator(cfg.latent_dim, cfg.data_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
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
    compute_V_DMD_Strict = {
        "kl_divergence": compute_V_DMD_kl_divergence,
        "sinkhorn": compute_V_DMD_Strict_sinkhorn,
        "sinkhorn_col": compute_V_DMD_Strict_sinkhorn_col,
        "sinkhorn_col_row": compute_V_DMD_Strict_sinkhorn_col_row,
    }[cfg.mmd_method]
    for epoch in range(cfg.epochs):
        if not is_running[0]: break

        # Gamma Linear Schedule
        curr_gamma = cfg.gamma_start + (cfg.gamma_end - cfg.gamma_start) * min(1.0, epoch / cfg.gamma_warmup_epochs)
        # Add temperature linear decay
        curr_T = cfg.T_start + (cfg.T_end - cfg.T_start) * min(1.0, epoch / cfg.T_warmup_epochs)

        # 1. Generate Samples
        e = torch.randn(cfg.N, cfg.latent_dim, device=cfg.device)
        x = model(e)
        
        # 2. Compute Target via Score-Matching Drift
        y_pos = sample_real_data(cfg.N, cfg.device)
        with torch.no_grad():
            V = compute_V_DMD_Strict(x, y_pos, x.clone(), curr_T, gamma =curr_gamma)
            x_target = x + V
        
        # 3. Optimization Step
        loss = F.mse_loss(x, x_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())

        # 4. Visualization
        if epoch % cfg.plot_interval == 0:
            # 1. Generate Samples
            e = torch.randn(cfg.N_plot, cfg.latent_dim, device=cfg.device)
            x = model(e)
            with torch.no_grad():
                V = compute_V_DMD_Strict(x, y_pos, x.clone(), curr_T, gamma =curr_gamma)

            if not update_plots(axs, fig, epoch, x, V, loss_history, curr_gamma, cfg):
                break
        # 5. Logging
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:06d} | Loss: {loss.item():.4e} | Gamma: {curr_gamma:.2f} | T: {curr_T:.2f}")
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