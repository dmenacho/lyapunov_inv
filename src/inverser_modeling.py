
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from lyap_models import LyapunovNet
from learners import DynamicSampler
import numpy as np
from tqdm import tqdm

class StabilityConstrainedNN:
    def __init__(
        self,
        lyapunov: LyapunovNet,
        resnet_dynamics: DynamicSampler,
        learning_rate: float,
        traj_steps: int,
        device: torch.device,
        alpha: float = 0.05,
        constraint_threshold: float = 1.0,
    ) -> None:
        self.lyapunov = lyapunov.to(device)
        self.lyapunov.eval()
        self.nn_dynamics = resnet_dynamics
        self.device = device
        self.learning_rate = learning_rate
        self.traj_steps = traj_steps
        self.alpha = alpha
        self.constraint_threshold = constraint_threshold

        self.x_mean: Optional[torch.Tensor] = None
        self.x_std: Optional[torch.Tensor] = None
        self.f_mean: Optional[torch.Tensor] = None
        self.f_std: Optional[torch.Tensor] = None

    def set_normalization_stats(self, x_mean, x_std, f_mean=None, f_std=None) -> None:
        self.x_mean = torch.as_tensor(x_mean, dtype=torch.float32, device=self.device)
        self.x_std  = torch.as_tensor(x_std,  dtype=torch.float32, device=self.device)
        if f_mean is not None and f_std is not None:
            self.f_mean = torch.as_tensor(f_mean, dtype=torch.float32, device=self.device)
            self.f_std  = torch.as_tensor(f_std,  dtype=torch.float32, device=self.device)

    def simulate_training_trajectory_with_gradients(self, w_init: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (trajectory, gradients) where:
           trajectory: (T+1, D)
           gradients:  (T, D)
        """
        traj_np, _, grads_np = self.nn_dynamics.simulate_trajectory(
            w_init=w_init, learning_rate=self.learning_rate, num_steps=self.traj_steps, return_gradients=True
        )
        assert grads_np is not None
        traj = torch.tensor(traj_np, dtype=torch.float32, device=self.device)
        grads = torch.tensor(grads_np, dtype=torch.float32, device=self.device)
        return traj, grads

    def evaluate_lyapunov_on_trajectory(self, trajectory: torch.Tensor, gradients: torch.Tensor) -> Tuple[float, float]:
        """
        Approximate constraints:
          c1 = max(0, -V(w0))  (positivity)
          c2 = max_t max(0, dV/dt + alpha*V)  (exponential stability-type condition)
        where f(w) = -∇L(w) (matching training collocation).
        """
        if self.x_mean is not None and self.x_std is not None:
            traj_n = (trajectory - self.x_mean) / (self.x_std + 1e-8)
            grads_n = gradients / (self.x_std + 1e-8)
        else:
            # fallback (older checkpoints)
            traj_n = (trajectory - trajectory.mean(dim=0)) / (trajectory.std(dim=0) + 1e-8)
            grads_n = gradients / (trajectory.std(dim=0) + 1e-8)

        V_vals: List[float] = []
        Vdot_vals: List[float] = []

        for t in range(traj_n.shape[0] - 1):
            state = traj_n[t : t + 1].requires_grad_(True)
            V = self.lyapunov(state)  # (1,1)
            V_vals.append(float(V.item()))
            V_grad = torch.autograd.grad(V.sum(), state, create_graph=False)[0]  # (1,D)

            # grad_raw = grads_n[t]
            grad_raw = gradients[t] # this is ∇L
            f_raw = -grad_raw # f = -∇L

            if self.f_mean is not None and self.f_std is not None:
                f_n = (f_raw - self.f_mean) / (self.f_std + 1e-8)
            else:
                f_n = f_raw 
            Vdot = (V_grad.squeeze(0) * f_n).sum()
            Vdot_vals.append(float(Vdot.item()))

        V_vals_np = np.asarray(V_vals, dtype=np.float32)
        Vdot_np = np.asarray(Vdot_vals, dtype=np.float32)

        c1 = max(0.0, -float(V_vals_np[0]))
        stability = Vdot_np + self.alpha * V_vals_np
        c2 = float(np.maximum(stability, 0.0).max())
        constraint = c1 + c2
        return float(V_vals_np.max()), constraint

    def evaluate_initialization(self, w0: np.ndarray) -> Tuple[float, float]:
        traj, grads = self.simulate_training_trajectory_with_gradients(w0)
        return self.evaluate_lyapunov_on_trajectory(traj, grads)

    def discover_stable_initializations(
        self,
        n_samples: int,
        weight_scale_range: Tuple[float, float],
        verbose_every: int = 25,
    ) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
        stable_weights: List[np.ndarray] = []
        metadata: List[Dict[str, float]] = []

        scales = np.random.uniform(weight_scale_range[0], weight_scale_range[1], n_samples)
        for i, scale in enumerate(tqdm(scales, desc="Weight init sampling")):
            w0 = (np.random.randn(self.nn_dynamics.state_dim).astype(np.float32) * float(scale))

            try:
                Vmax, viol = self.evaluate_initialization(w0)

                if (i + 1) % max(1, verbose_every) == 0:
                    print(f"[{i+1}/{n_samples}] scale={scale:.4f}  Vmax={Vmax:.4f}  viol={viol:.4f}")

                if viol < self.constraint_threshold:
                    stable_weights.append(w0)
                    metadata.append(
                        {
                            "initial_scale": float(scale),
                            "V_max": float(Vmax),
                            "constraint_violation": float(viol),
                            "weight_norm": float(np.linalg.norm(w0)),
                        }
                    )
            except Exception as e:
                # keep going
                if (i + 1) % max(1, verbose_every) == 0:
                    print(f"[{i+1}/{n_samples}] error: {e}")
                continue

        return stable_weights, metadata