import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from pinn_lyap_nn import LyapunovNet, NNDynamics
from typing import Tuple, List, Dict

class StabilityConstrained:
    def __init__(self, lyapunov, nn_dynamics, learning_rate= 0.1, device = 'cpu'):
        self.lyapunov = lyapunov.to(device)
        self.lyapunov.eval()
        
        self.nn_dynamics = nn_dynamics
        self.device = device
        self.learning_rate = learning_rate        
        self.num_steps = 15
        self.dt = 1.0 / learning_rate

    def simulate_training_trajectory(self, w_init, num_steps):
        if num_steps is None:
            num_steps = self.num_steps

        model = nn.Sequential(
            nn.Linear(self.nn_dynamics.input_dim, self.nn_dynamics.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.nn_dynamics.hidden_dim, self.nn_dynamics.output_dim)
        ).to(self.device)

        self.nn_dynamics.unflatten_params(w_init, model)
        
        trajectory = [torch.tensor(w_init, dtype=torch.float32, device=self.device)]
        
        loss_fn = nn.MSELoss()

        for step in range(num_steps):
            y_pred = model(self.nn_dynamics.X_train_tensor)
            loss = loss_fn(y_pred, self.nn_dynamics.y_train_tensor)

            loss.backward()
            
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.data -= self.learning_rate * param.grad
                        param.grad.zero_()
            
            # Record state
            w_current = self.nn_dynamics.flatten_params(model)
            trajectory.append(
                torch.tensor(w_current, dtype=torch.float32, device=self.device)
            )
        
        trajectory = torch.stack(trajectory)
        return trajectory
    
    def compute_loss( self, w0):
        w0_np = w0.detach().cpu().numpy()
        trajectory = self.simulate_training_trajectory(w0_np, num_steps=self.num_steps)
        
        trajectory_normalized = (trajectory - trajectory.mean(dim=0)) / (trajectory.std(dim=0) + 1e-8)
        final_loss_value = trajectory[-1].norm()**2
        L_data = final_loss_value
        
        V_values = []
        V_dot_values = []
        
        for i in range(trajectory_normalized.shape[0]):
            state = trajectory_normalized[i:i+1].clone().detach().requires_grad_(True)
            V_val = self.lyapunov(state)
            V_values.append(V_val)

            V_grad = torch.autograd.grad(V_val.sum(), state, create_graph=True)[0]

            if i < trajectory_normalized.shape[0] - 1:
                f_val = (trajectory_normalized[i+1] - trajectory_normalized[i]) / self.dt
            else:
                f_val = (trajectory_normalized[i] - trajectory_normalized[i-1]) / self.dt
            
            V_dot = (V_grad * f_val).sum()
            V_dot_values.append(V_dot)

        V_vals = torch.stack(V_values)
        V_dot_vals = torch.stack(V_dot_values)
        
        w0_normalized = (w0 - w0.mean()) / (w0.std() + 1e-8)
        w0_normalized = w0_normalized.unsqueeze(0)
        roa_constraint_1 = torch.relu(self.lyapunov(w0_normalized))
        
        # Decay constraint: V̇ ≤ -α·V
        alpha = 0.05
        roa_constraint_2 = torch.relu(V_dot_vals + alpha * V_vals.squeeze()).mean()
        
        L_lyapunov = roa_constraint_1 + roa_constraint_2

        total_loss = L_lyapunov
        
        return total_loss, L_data, L_lyapunov
    
    def optimize_initialization(self, w0_init, learning_rate_opt, max_iters):
        w0 = nn.Parameter(torch.tensor(w0_init, dtype=torch.float32, device=self.device))
        optimizer = optim.Adam([w0], lr=learning_rate_opt)

        prev_loss = float('inf')
        patience_counter = 0
        patience = 5

        for iteration in range(max_iters):

            total_loss, loss_data, loss_lyapunov = self.compute_loss(w0)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([w0], max_norm=1.0)
            optimizer.step()
            
            # # Early stopping
            # if abs(total_loss.item() - prev_loss) < tol:
            #     patience_counter += 1
            #     if patience_counter > patience:
            #         print(f"  Converged at iteration {iteration+1}")
            #         break
            # else:
            #     patience_counter = 0
            
            # prev_loss = total_loss.item()
            
            if (iteration + 1) % 10 == 0:
                print(f"  Iter {iteration+1}: Loss={total_loss.item():.4f}, "
                      f"L_data={loss_data.item():.4f}, L_lyap={loss_lyapunov.item():.4f}")
                
        total_loss, loss_data, loss_lyapunov = self.compute_loss(w0)
        
        return w0.detach().cpu().numpy(), total_loss.item(), loss_data.item(), loss_lyapunov.item()

    def discover_stable_initializations( self, n_samples, weight_scale_range):
        stable_weights = []
        metadata = []

        weight_scales = np.random.uniform(*weight_scale_range, n_samples)
        
        for i, scale in enumerate(tqdm(weight_scales, desc="Weight Initialization Optimization")):
            w0_init = np.random.randn(self.nn_dynamics.state_dim).astype(np.float32) * scale
            
            print(f"\n[{i+1}/{n_samples}] Weight scale = {scale:.4f}")
 
            w0_opt, loss_total, loss_data, loss_lyapunov = self.optimize_initialization(
                w0_init,
                learning_rate_opt=1e-4,
                max_iters=25,
            )
            
            print(f"Total Loss = {loss_total:.4f}, Data Loss = {loss_data:.4f}, Lyapunov Loss = {loss_lyapunov:.4f}")

            if loss_total <= 0.9:
                stable_weights.append(w0_opt)
                metadata.append({
                    'initial_scale': float(scale),
                    'loss_total': float(loss_total),
                    'loss_data': float(loss_data),
                    'loss_lyapunov': float(loss_lyapunov),
                    'weight_norm': float(np.linalg.norm(w0_opt))
                })
        
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
    
    loss_totals = np.array([m['loss_total'] for m in metadata])
    print(f"\nTotal Loss:")
    print(f"  Mean:   {loss_totals.mean():.6f}")
    print(f"  Std:    {loss_totals.std():.6f}")
    print(f"  Min:    {loss_totals.min():.6f}")
    print(f"  Max:    {loss_totals.max():.6f}")
    

    for d in range(min(10, state_dim)):
        w_d = stable_weights_array[:, d]
        print(f"  w[{d}]: mean={w_d.mean():.6f}, std={w_d.std():.6f}, "
              f"range=[{w_d.min():.6f}, {w_d.max():.6f}]")
    
    # Pairwise distance between stable weights (diversity)
    # if len(stable_weights) > 1:
    #     distances = []
    #     for i in range(len(stable_weights)):
    #         for j in range(i+1, len(stable_weights)):
    #             dist = np.linalg.norm(stable_weights[i] - stable_weights[j])
    #             distances.append(dist)
        
    #     distances = np.array(distances)
    #     print(f"\nWeight Configuration Diversity (pairwise distances):")
    #     print(f"  Mean distance:   {distances.mean():.6f}")
    #     print(f"  Std distance:    {distances.std():.6f}")
    #     print(f"  Min distance:    {distances.min():.6f}")
    #     print(f"  Max distance:    {distances.max():.6f}")


def plot_stable_weights( metadata, criterion):
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
    
    # 2. Loss components scatter
    loss_data = np.array([m['loss_data'] for m in metadata])
    loss_lyap = np.array([m['loss_lyapunov'] for m in metadata])
    axes[0, 1].scatter(loss_data, loss_lyap, alpha=0.6, s=100, color='coral')
    axes[0, 1].set_xlabel('Data Loss', fontsize=12)
    axes[0, 1].set_ylabel('Lyapunov Loss', fontsize=12)
    axes[0, 1].set_title('Trade-off: Data Loss vs Stability Loss', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Initial scale vs final weight norm
    initial_scales = np.array([m['initial_scale'] for m in metadata])
    axes[1, 0].scatter(initial_scales, weight_norms, alpha=0.6, s=100, color='green')
    axes[1, 0].set_xlabel('Initial Scale', fontsize=12)
    axes[1, 0].set_ylabel('Final Weight Norm', fontsize=12)
    axes[1, 0].set_title('Initialization Scale → Final Weight Configuration', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # 4. Loss distribution
    loss_totals = np.array([m['loss_total'] for m in metadata])
    axes[1, 1].hist(loss_totals, bins=15, alpha=0.7, edgecolor='black', color='mediumpurple')
    axes[1, 1].set_xlabel('Total Loss', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Distribution of Total Loss\n(for stable configurations)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'stable_weight_initializations_{criterion}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {filename}")
    plt.close()


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    nn_dynamics = NNDynamics(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        device=device
    )
    
    lyapunov = LyapunovNet(state_dim=nn_dynamics.state_dim, hidden_dim=128).to(device)
    lyapunov.load_state_dict(torch.load('models/lyapunov_nn_training.pt', map_location=device))

    solver = StabilityConstrained(
        lyapunov,
        nn_dynamics,
        learning_rate=0.1,
        device=device
    )

    stable_weights, metadata = solver.discover_stable_initializations(
        n_samples=1000,
        weight_scale_range=(1e-3, 0.5)
    )
    
    print(f" Found {len(stable_weights)} stable weight configurations")
    
    analyze_stable_weights(stable_weights, metadata, nn_dynamics.state_dim)
    Path('results').mkdir(exist_ok=True)
    np.save('results/stable_weights.npy', np.array(stable_weights))

    metadata_array = np.array([ [m['initial_scale'], m['loss_total'], m['loss_data'], m['loss_lyapunov'], m['weight_norm']] for m in metadata])
    np.save('results/stable_weights_metadata.npy', metadata_array)

    plot_stable_weights(metadata, criterion="stable_initializations")

if __name__ == "__main__":
    main()