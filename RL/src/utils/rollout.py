from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union
from utils.common import angle_normalize

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class RolloutData:
    timesteps: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    actions: List[float] = field(default_factory=list)
    theta: List[float] = field(default_factory=list)
    theta_dot: List[float] = field(default_factory=list)
    phi: List[float] = field(default_factory=list)
    u_cmd: List[float] = field(default_factory=list)

    def add(self, timestep: int, action: float, reward: float, info: Dict[str, Any]) -> None:
        self.timesteps.append(int(timestep))
        self.actions.append(float(action))
        self.rewards.append(float(reward))
        self.theta.append(float(info.get('theta', 0.0)))
        self.theta_dot.append(float(info.get('theta_dot', 0.0)))
        self.phi.append(float(info.get('phi', 0.0)))
        self.u_cmd.append(float(info.get('u_cmd', action)))

    def add_separator(self, timestep: int) -> None:
        self.timesteps.append(int(timestep))
        self.actions.append(float("nan"))
        self.rewards.append(float("nan"))
        self.theta.append(float("nan"))
        self.theta_dot.append(float("nan"))
        self.phi.append(float("nan"))
        self.u_cmd.append(float("nan"))


def plot_rollout(data: RolloutData, out_path: Union[str, Path]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    theta = np.asarray(data.theta, dtype=float) if data.theta else np.asarray([])
    theta = theta + np.pi if theta.size else np.asarray([])
    theta = angle_normalize(theta)

    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

    axes[0].plot(data.timesteps, theta, color='tab:blue', linewidth=1)
    axes[0].set_ylabel('theta (unwrapped)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(data.timesteps, data.theta_dot, color='tab:orange', linewidth=1)
    axes[1].set_ylabel('theta_dot')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(data.timesteps, data.phi, color='tab:green', linewidth=1)
    axes[2].set_ylabel('phi')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(data.timesteps, data.actions, color='tab:purple', linewidth=1)
    axes[3].set_ylabel('action')
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(data.timesteps, data.rewards, color='tab:red', linewidth=1)
    axes[4].set_ylabel('reward')
    axes[4].set_xlabel('timestep')
    axes[4].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path