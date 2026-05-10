import torch
import scipy.io
import numpy as np
from pathlib import Path
from utils.custom_paths import MODELS_DIR

from nets.a2c import ModelA2C

if __name__ == "__main__":
    obs_size = 5
    act_size = 1
    model_hid_size = 64

    model = ModelA2C(obs_size=obs_size, act_size=act_size, hid_size=model_hid_size)

    save_path = MODELS_DIR / "rl_ppo_3" / "ppo_pendulum_model.pth"
    checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    model.eval()
    print("Dostępne warstwy w modelu:")
    for name, param in model.named_parameters():
        print(f"- {name} (wymiary: {param.shape})")

    weights_dict = {}
    for name, param in model.named_parameters():
        if 'actor_net' in name:
            arr = param.detach().cpu().numpy().astype(np.float32)
            if 'bias' in name:
                arr = arr.reshape(-1, 1)
            clean_name = name.replace('.', '_')
            weights_dict[clean_name] = arr

    scipy.io.savemat( MODELS_DIR / "rl_ppo_3" / 'ppo_actor_weights.mat', weights_dict)
    print("\nWagi wyeksportowane pomyślnie do ppo_actor_weights.mat!")