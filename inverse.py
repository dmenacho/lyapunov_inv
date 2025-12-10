import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

class DoubleSpringSystem:
    def __init__(self):
        self.Tx = 0.0
        self.Ty = 2.0
        self.R1 = 1.0 
        self.R2 = 1.0
        self.g = 9.8
    
    def dynamics(self, state, t, theta):
        u1x, u1y, u2x, u2y, v1x, v1y, v2x, v2y = state
        m1, m2, k1, k2, b1, b2 = theta
        
        # SPRING 1
        dx1 = u1x - self.Tx
        dy1 = u1y - self.Ty
        L1 = np.sqrt(dx1**2 + dy1**2)
        L1 = max(L1, 1e-8)
        difference_S1 = L1 - self.R1
        ux1 = dx1 / L1
        uy1 = dy1 / L1
        F_spring1_x = -k1 * difference_S1 * ux1
        F_spring1_y = -k1 * difference_S1 * uy1
        
        # SPRING 2
        dx2 = u2x - u1x
        dy2 = u2y - u1y
        L2 = np.sqrt(dx2**2 + dy2**2)
        L2 = max(L2, 1e-8)
        difference_S2 = L2 - self.R2
        ux2 = dx2 / L2
        uy2 = dy2 / L2
        F_spring2_x_on_1 = k2 * difference_S2 * ux2
        F_spring2_y_on_1 = k2 * difference_S2 * uy2
        F_spring2_x = -F_spring2_x_on_1
        F_spring2_y = -F_spring2_y_on_1
        
        # DAMPING
        damping1_x = -b1 * v1x
        damping1_y = -b1 * v1y
        damping2_x = -b2 * v2x
        damping2_y = -b2 * v2y
        
        # GRAVITY
        g_1y = -m1 * self.g
        g_2y = -m2 * self.g
        
        # SUM FORCES
        Total_F1x = F_spring1_x + F_spring2_x_on_1 + damping1_x
        Total_F1y = F_spring1_y + F_spring2_y_on_1 + damping1_y + g_1y
        a1x = Total_F1x / m1
        a1y = Total_F1y / m1
        Total_F2x = F_spring2_x + damping2_x
        Total_F2y = F_spring2_y + damping2_y + g_2y
        a2x = Total_F2x / m2
        a2y = Total_F2y / m2
        
        return [v1x, v1y, v2x, v2y, a1x, a1y, a2x, a2y]
    
class ParametricSurrogate(nn.Module):
    def __init__(self, state_dim=8, param_dim=6, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + param_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
    def forward(self, x0, theta, t):
        x_in = torch.cat([x0, theta, t], dim=1)
        return self.net(x_in)

class LyapunovNet(nn.Module):
    def __init__(self, state_dim=8, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, V):
        V = self.net(V)
        V = torch.relu(V) + 1e-6 
        return V
    
class StabilityConstrainedInverseSolver:
    def __init__(self, surrogate, lyapunov, system, device='cuda'):
        self.surrogate = surrogate.to(device)
        self.surrogate.eval()
        self.lyapunov = lyapunov.to(device)
        self.lyapunov.eval()
        self.system = system
        self.device = device
        self.t_times = np.linspace(0, 15, 150)
        self.t_tensor = torch.tensor(self.t_times, dtype=torch.float32, device=device).unsqueeze(1)
        
        # Time step for finite differences
        self.dt = self.t_times[1] - self.t_times[0]
    
    def compute_loss(self, x0, theta):
        x0_batch = x0.unsqueeze(0).repeat(len(self.t_times), 1)
        theta_batch = theta.unsqueeze(0).repeat(len(self.t_times), 1)
        
        x_pred = self.surrogate(x0_batch, theta_batch, self.t_tensor)
        x_final = x_pred[-1] 
        v_final = x_final[4:8] 

        L_data = torch.norm(v_final) ** 2
        
        V_values = []
        V_dot_values = []
        
        for i in range(len(x_pred)):
            state = x_pred[i].clone().detach().requires_grad_(True)
            V_val = self.lyapunov(state)
            V_grad = torch.autograd.grad(V_val.sum(), state, create_graph=True)[0]
            if i < len(x_pred) - 1:
                f_val = (x_pred[i+1] - x_pred[i]) / self.dt
            else:
                f_val = (x_pred[i] - x_pred[i-1]) / self.dt

            V_dot = (V_grad * f_val).sum()
            
            V_values.append(V_val)
            V_dot_values.append(V_dot)
        
        V_vals = torch.stack(V_values)
        V_dot_vals = torch.stack(V_dot_values)
        roa_constraint_1 = torch.relu(-self.lyapunov(x0.unsqueeze(0)))

        alpha = 0.1 
        roa_constraint_2 = torch.relu(V_dot_vals + alpha * V_vals).mean()
        L_lyapunov = roa_constraint_1 + roa_constraint_2
        
        total_loss = L_data + 5.0 * L_lyapunov
        return total_loss, L_data, L_lyapunov, v_final

    
    def optimize_single(self, x0_init, theta_init, learning_rate=1e-3, max_iters=500, tol=1e-4):
        x0 = nn.Parameter(x0_init.clone().detach().to(self.device))
        theta = nn.Parameter(theta_init.clone().detach().to(self.device))
        
        optimizer = optim.Adam([x0, theta], lr=learning_rate)

        for _ in range(max_iters):
            loss, L_data, _, v_final = self.compute_loss(x0, theta)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([x0, theta], max_norm=1.0)
            optimizer.step()
            
        loss, L_data, _, v_final = self.compute_loss(x0, theta)
        final_speed = torch.norm(v_final).item()
        return theta.detach().cpu().numpy(), loss.item(), L_data.item(), final_speed
    
    def discover_admissible_set(self, n_samples=100, device='cuda'):
        admissible_thetas_loss_total = []
        admissible_thetas_loss_data = []
        thetas = np.random.uniform(
            low=[0.3, 0.3, 2.0, 2.0, -0.5, -0.5],
            high=[2.0, 2.0,  8.0, 8.0, 0.5, 0.5],
            size=(n_samples, 6)
        )
        
        x0_nominal = np.array([0.25, 0.5, -0.2, -1.0, 0.0, 0.0, 0.0, 0.0])
        
        for i, theta in enumerate(tqdm(thetas, desc="Inverse Optimization")):
            x0_init = torch.tensor(x0_nominal, dtype=torch.float32)
            theta_init = torch.tensor(theta, dtype=torch.float32) 
            print(theta_init)
            theta_opt, final_loss, loss_data, V_max = self.optimize_single(x0_init, theta_init,learning_rate=1e-3,max_iters=10,tol=1e-4)
            print(f"Losses => Total: {final_loss:.4f}, L_data: {loss_data:.4f}, V_max: {V_max:.4f}")
            print(theta_opt)
            if final_loss < 1.0:
                admissible_thetas_loss_total.append(theta_opt)
            if loss_data < 1.0:
                admissible_thetas_loss_data.append(theta_opt)

        admissible_thetas_total = np.array(admissible_thetas_loss_total)
        admissible_thetas_data = np.array(admissible_thetas_loss_data)
        return admissible_thetas_total, admissible_thetas_data


def plot_admissible_set(admissible_thetas):
    param_names = ['m1','m2','k1','k2','b1','b2']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, name in enumerate(param_names):
        axes[i].hist(admissible_thetas[:, i], bins=20, alpha=0.7, edgecolor='black', color='purple')
        axes[i].set_xlabel(name, fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
        axes[i].set_title(f'Distribution of {name}', fontsize=12)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('admissible_parameter_distributions.png', dpi=150)
    plt.show()
   
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    system = DoubleSpringSystem()

    surrogate_path = 'models/parametric_surrogate.pt'
    surrogate = ParametricSurrogate(state_dim=8, param_dim=6, hidden_dim=256).to(device)
    
    surrogate.load_state_dict(torch.load(surrogate_path, map_location=device))

    lyapunov = LyapunovNet(state_dim=8, hidden_dim=128).to(device)
    lyapunov.load_state_dict(torch.load('models/lyapunov_zubov.pt', map_location=device))

    solver = StabilityConstrainedInverseSolver(surrogate, lyapunov, system, device=device)
    total, data = solver.discover_admissible_set(n_samples=1000)

    plot_admissible_set(total)
    np.save('admissible_thetas_total.npy', total)
    plot_admissible_set(data)
    np.save('admissible_thetas_data.npy', data)