import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from pinn_lyap_nn import LyapunovNet, NNDynamics, ResNetDynamics, TinyResNet
from typing import Tuple, List, Dict

class StabilityConstrainedNN:
    def __init__(self, lyapunov, resnet_dynamics, learning_rate, device='cpu'):
        self.lyapunov = lyapunov.to(device)
        self.lyapunov.eval()
    
        self.nn_dynamics = resnet_dynamics
        self.device = device
        self.learning_rate = learning_rate      
        self.num_steps = 50
        # self.dt = 0.01
        
        self.x_mean = None
        self.x_std = None

    def set_normalization_stats(self, x_mean, x_std):
        self.x_mean = torch.tensor(x_mean, dtype=torch.float32, device=self.device)
        self.x_std = torch.tensor(x_std, dtype=torch.float32, device=self.device)

    def simulate_training_trajectory(self, w_init, num_steps):
        """Returns trajectory AND gradients"""
        model = TinyResNet().to(self.device)
        self.nn_dynamics.unflatten_params(w_init, model)
        
        trajectory = [torch.tensor(w_init, dtype=torch.float32, device=self.device)]
        gradients = []  # NEW: store gradients
        
        loss_fn = nn.CrossEntropyLoss()
        
        for step in range(num_steps):
            batch_idx = np.random.randint(0, len(self.nn_dynamics.X_train))
            X_batch = self.nn_dynamics.X_train[batch_idx]
            y_batch = self.nn_dynamics.y_train[batch_idx]
            
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            
            # Store gradient BEFORE updating
            grad_flat = torch.cat([p.grad.flatten() for p in model.parameters()])
            gradients.append(grad_flat.detach().clone())
            
            with torch.no_grad():
                for param in model.parameters():
                    param.data -= self.learning_rate * param.grad
                    param.grad.zero_()
            
            w_current = self.nn_dynamics.flatten_params(model)
            trajectory.append(torch.tensor(w_current, dtype=torch.float32, device=self.device))
        
        trajectory = torch.stack(trajectory)  # (T+1, D)
        gradients = torch.stack(gradients)    # (T, D)
        return trajectory, gradients

    def evaluate_lyapunov_on_trajectory(self, w0, trajectory, gradients):
        # Normalize trajectory consistently
        if self.x_mean is not None:
            trajectory_normalized = (trajectory - self.x_mean) / (self.x_std + 1e-8)
            gradients_normalized = gradients / (self.x_std + 1e-8)
        else:
            trajectory_normalized = (trajectory - trajectory.mean(dim=0)) / (trajectory.std(dim=0) + 1e-8)
            gradients_normalized = gradients / (trajectory.std(dim=0) + 1e-8)
        
        V_values = []
        V_dot_values = []
        
        # Evaluate V and dV/dt along trajectory
        for i in range(trajectory_normalized.shape[0] - 1):
            # Compute V(w_i)
            state = trajectory_normalized[i:i+1].requires_grad_(True)
            V_val = self.lyapunov(state)
            V_values.append(V_val.item())
            
            # Compute gradien
            V_grad = torch.autograd.grad(V_val.sum(), state, create_graph=False)[0]
            
            # Compute gradient descent dynamics
            f_normalized = -gradients_normalized[i]
            V_dot = (V_grad * f_normalized).sum().item()
            V_dot_values.append(V_dot)
        
        V_values = np.array(V_values)
        V_dot_values = np.array(V_dot_values)
        
        c1 = max(0, -V_values[0])  # max(0, -V(w0))
        
        alpha = 0.05
        stability_constraint = V_dot_values + alpha * V_values
        c2 = np.maximum(stability_constraint, 0).max()  # Worst-case violation
        
        print(f"c1 (positivity): {c1:.6f}, c2 (stability): {c2:.6f}")
        constraint = c1 + c2
        
        return V_values.max(), constraint

    
    def evaluate_initialization(self, w0_init):
        w0_np = np.asarray(w0_init).ravel().astype(np.float32)
        
        trajectory, gradient = self.simulate_training_trajectory(w0_np, num_steps=self.num_steps)
        
        V_max, constraint = self.evaluate_lyapunov_on_trajectory(w0_np, trajectory, gradient)
        return V_max, constraint

    def discover_stable_initializations(self, n_samples, weight_scale_range):
        stable_weights = []
        metadata = []

        weight_scales = np.random.uniform(*weight_scale_range, n_samples)
        
        for i, scale in enumerate(tqdm(weight_scales, desc="Weight Initialization Sampling")):
            w0_init = np.random.randn(self.nn_dynamics.state_dim).astype(np.float32) * scale
            
            print(f"\n[{i+1}/{n_samples}] Weight scale = {scale:.4f}")
            
            try:
                V_max, constraint_viol = self.evaluate_initialization(w0_init)
                
                print(f"  V_max = {V_max:.4f}, Constraint violation = {constraint_viol:.4f}")

                if constraint_viol < 1:
                    stable_weights.append(w0_init)
                    metadata.append({
                        'initial_scale': float(scale),
                        'V_max': float(V_max),
                        'constraint_violation': float(constraint_viol),
                        'weight_norm': float(np.linalg.norm(w0_init))
                    })
                    
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        return stable_weights, metadata
    
def analyze_stable_weights(stable_weights, metadata, state_dim):
    if not stable_weights:
        print("No stable weights found")
        return
    
    stable_weights_array = np.array(stable_weights)
    
    weight_norms = np.array([m['weight_norm'] for m in metadata])
    print(f"\nWeight Norm (||w₀||):")
    print(f"  Mean:   {weight_norms.mean():.6f}")
    print(f"  Std:    {weight_norms.std():.6f}")
    print(f"  Min:    {weight_norms.min():.6f}")
    print(f"  Max:    {weight_norms.max():.6f}")
    print(f"  Median: {np.median(weight_norms):.6f}")
    
    V_max_vals = np.array([m['V_max'] for m in metadata])
    print(f"\nLyapunov V_max:")
    print(f"  Mean:   {V_max_vals.mean():.6f}")
    print(f"  Std:    {V_max_vals.std():.6f}")
    print(f"  Min:    {V_max_vals.min():.6f}")
    print(f"  Max:    {V_max_vals.max():.6f}")
    
    constraint_viols = np.array([m['constraint_violation'] for m in metadata])
    print(f"\nConstraint Violations:")
    print(f"  Mean:   {constraint_viols.mean():.6f}")
    print(f"  Std:    {constraint_viols.std():.6f}")
    print(f"  Min:    {constraint_viols.min():.6f}")
    print(f"  Max:    {constraint_viols.max():.6f}")

    for d in range(min(10, state_dim)):
        w_d = stable_weights_array[:, d]
        print(f"  w[{d}]: mean={w_d.mean():.6f}, std={w_d.std():.6f}, "
              f"range=[{w_d.min():.6f}, {w_d.max():.6f}]")


def plot_stable_weights(metadata, criterion):
    if not metadata:
        print("No metadata to plot!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Weight norm distribution
    weight_norms = np.array([m['weight_norm'] for m in metadata])
    axes[0, 0].hist(weight_norms, bins=15, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0, 0].set_xlabel('Weight Norm (||w₀||)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Distribution of Stable Weight Initializations\n(by norm)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. V_max vs Constraint violation
    V_max_vals = np.array([m['V_max'] for m in metadata])
    constraint_viols = np.array([m['constraint_violation'] for m in metadata])
    axes[0, 1].scatter(V_max_vals, constraint_viols, alpha=0.6, s=100, color='coral')
    axes[0, 1].set_xlabel('V_max', fontsize=12)
    axes[0, 1].set_ylabel('Constraint Violation', fontsize=12)
    axes[0, 1].set_title('Lyapunov Value vs Constraint', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Initial scale vs final weight norm
    initial_scales = np.array([m['initial_scale'] for m in metadata])
    axes[1, 0].scatter(initial_scales, weight_norms, alpha=0.6, s=100, color='green')
    axes[1, 0].set_xlabel('Initial Scale', fontsize=12)
    axes[1, 0].set_ylabel('Final Weight Norm', fontsize=12)
    axes[1, 0].set_title('Initialization Scale → Final Weight Configuration', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # 4. Constraint violation distribution
    axes[1, 1].hist(constraint_viols, bins=15, alpha=0.7, edgecolor='black', color='mediumpurple')
    axes[1, 1].set_xlabel('Constraint Violation', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Distribution of Constraint Violations\n(for stable configurations)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'stable_weight_initializations_{criterion}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {filename}")
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    resnet_dynamics = ResNetDynamics(device=device, num_train_samples=10000)
    
    lyapunov = LyapunovNet(state_dim=resnet_dynamics.state_dim, hidden_dim=256).to(device)
    lyapunov.load_state_dict(torch.load('/Users/daniel/Codes/MS Michigan/AI for Science/lyapunov_inv/training_info/models/1000_50_0.01/20260202_121102/lyapunov_resnet.pt', map_location=device))

    solver = StabilityConstrainedNN(lyapunov, resnet_dynamics, learning_rate=1e-2)

    stable_weights, metadata = solver.discover_stable_initializations(
        n_samples=1000,  # Start with fewer samples for testing
        weight_scale_range=(0.1, 1)
    )
    
    print(f"\nFound {len(stable_weights)} stable weight configurations")
    
    analyze_stable_weights(stable_weights, metadata, resnet_dynamics.state_dim)
    
    Path('results').mkdir(exist_ok=True)
    if len(stable_weights) > 0:
        np.save('results/stable_weights.npy', np.array(stable_weights))
        metadata_array = np.array([
            [m['initial_scale'], m['V_max'], m['constraint_violation'], m['weight_norm']] 
            for m in metadata
        ])
        np.save('results/stable_weights_metadata.npy', metadata_array)
        plot_stable_weights(metadata, criterion="stable_initializations")


if __name__ == "__main__":
    main()