import numpy as np
import matplotlib.pyplot as plt
from reac_wheel_sim.reaction_wheel_env import *
from scipy.io import loadmat
from config import LOGS_DIR

import pytest
import logging

# logging = logging.getlogging(__name__)


def _unwrap_scalar_object_array(value):
    while isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        value = value.item()
    return value

def load_matlab_data(file_path, start_sample=0):
    raw_data = loadmat(file_path, squeeze_me=True)
    sd = raw_data['StateData']
    time_full = _unwrap_scalar_object_array(sd['time'])
    time = np.asarray(time_full, dtype=np.float64)[start_sample:]
    time = time - time[0]
    signals = _unwrap_scalar_object_array(sd['signals'])

    def get_val(idx):
        val = _unwrap_scalar_object_array(signals[idx]['values'])
        return np.asarray(val, dtype=np.float64)[start_sample:]

    data_dict = {
        't': time,
        'u':          get_val(0), 
        'pendPosZD':  get_val(1), 
        'pendPosZU':  get_val(2), 
        'pendVel':    get_val(3), 
        'diskPos':    get_val(4), 
        'diskVel':    get_val(5)  
    }

    logging.info(f"Pomyślnie wczytano dane: {len(time)} próbek.")
    logging.info(f"Długość u: {len(data_dict['u'])}")
    
    return data_dict


def compare_real_vs_sim(mat_file_path, start_sample=0, save_path=None, show=False):
    real_data = load_matlab_data(mat_file_path, start_sample=start_sample)
    u_real = real_data["u"]
    t_real = real_data["t"]

    env = ReactionWheelEnv()

    initial_state = np.array(
        [real_data["pendPosZD"][0], real_data["pendVel"][0], real_data["diskVel"][0]],
        dtype=np.float32,
    )

    env.reset(seed=0)
    env.state = initial_state.copy()
    env.prev_u = float(u_real[0])
    env.step_count = 0
    env.max_episode_steps = max(env.max_episode_steps, len(u_real) + 1)

    # Model parameters
    env.K_sin = -27.311296
    env.K_pend_vel = -0.059984
    env.K_reac_wheel = -0.010594

    sim_theta = []
    sim_theta_dot = []
    sim_disk_vel = []
    ctrl_vec = []

    for u in u_real:
        obs, _, terminated, truncated, _ = env.step(np.array([u], dtype=np.float32))

        theta = np.arctan2(obs[0], obs[1])
        ctrl_vec.append(obs[4])
        sim_theta.append(theta)
        sim_theta_dot.append(obs[2])
        sim_disk_vel.append(obs[3])

        if terminated or truncated:
            logging.info("Symulacja zakończona wcześniej przez środowisko gym.")
            break

    sim_theta = np.array(sim_theta)
    sim_theta_dot = np.array(sim_theta_dot)
    sim_disk_vel = np.array(sim_disk_vel)
    n = len(sim_theta)
    t_real = t_real[:n]
    
    real_data["pendPosZD"] = real_data["pendPosZD"][:n]
    real_data["pendPosZU"] = real_data["pendPosZU"][:n]
    real_data["pendVel"] = real_data["pendVel"][:n]
    real_data["diskVel"] = real_data["diskVel"][:n]
    real_data["u"] = real_data["u"][:n]

    # Obliczanie błędu (RMSE)
    rmse_theta = np.sqrt(np.mean((real_data["pendPosZD"] - sim_theta) ** 2))
    logging.info(f"RMSE dla kąta wahadła: {rmse_theta:.4f} rad")

    # 5. Wyświetlanie wyników
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Wykres pozycji wahadła
    axs[0].plot(t_real, real_data["pendPosZD"], "k--", label="MATLAB (Real)", alpha=0.8)
    axs[0].plot(t_real, sim_theta, "b-", label="Python (Sim)", alpha=0.5)
    axs[0].set_ylabel("Kąt wahadła [rad]")
    axs[0].set_title(f"Porównanie modelu dla pliku: {mat_file_path}")
    axs[0].legend()
    axs[0].grid(True)

    # Wykres prędkości wahadła
    axs[1].plot(t_real, real_data["pendVel"], "k--", alpha=0.8)
    axs[1].plot(t_real, sim_theta_dot, "b-", alpha=0.5)
    axs[1].set_ylabel("Prędkość wahadła [rad/s]")
    axs[1].grid(True)

    # Wykres prędkości koła
    axs[2].plot(t_real, real_data["diskVel"], "k--", alpha=0.8)
    axs[2].plot(t_real, sim_disk_vel, "b-", alpha=0.5)
    axs[2].set_ylabel("Prędkość koła [rad/s]")
    axs[2].set_xlabel("Czas [s]")
    axs[2].grid(True)

    # Wykres prędkości koła
    axs[3].plot(t_real, real_data["u"], "k--", alpha=0.8)
    axs[3].plot(t_real, ctrl_vec, "b-", alpha=0.5)
    axs[3].set_ylabel("Prędkość koła [rad/s]")
    axs[3].set_xlabel("Czas [s]")
    axs[3].grid(True)

    if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)

    return rmse_theta

def test_env_sim():
    rmse_theta = compare_real_vs_sim(
        "/home/igorsiata/studia/Reaction-Wheel-Pendulum/data/14_04/ident_ster_bez_ciezarka.mat",
        start_sample=10,
        save_path=LOGS_DIR / "pytest" / "test_env_sim_real_data.png",
        show=False
    )
    logging.info(rmse_theta)
    assert rmse_theta < 10
