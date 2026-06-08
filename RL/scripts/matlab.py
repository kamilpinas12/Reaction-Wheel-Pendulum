import argparse
from pathlib import Path

import numpy as np
import torch
import scipy.io as sio
from stable_baselines3 import TD3

def extract_to_matlab(checkpoint_path: Path, output_dir: Path) -> None:
    print(f"Loading model from: {checkpoint_path}")
    
    # Load the TD3 model on CPU
    model = TD3.load(str(checkpoint_path), device="cpu")

    # Access the actor network
    actor = model.actor

    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dict = {}

    print(f"\nExtracting Actor matrices for MATLAB to: {output_dir.resolve()}")
    print("-" * 50)
    
    # Iterate over all parameters (weights and biases) in the actor network
    for name, param in actor.named_parameters():
        # Detach from computational graph and convert to NumPy array
        param_np = param.detach().numpy()
        
        # MATLAB variable names cannot contain dots. 
        # Replace '.' with '_' (e.g., 'mu.0.weight' becomes 'mu_0_weight')
        safe_name = name.replace(".", "_")
        
        # Add to our dictionary
        weights_dict[safe_name] = param_np
        
        print(f"Extracted {safe_name: <20} | Shape: {str(param_np.shape): <15}")

    # Save the dictionary as a .mat file
    mat_path = output_dir / "actor_weights.mat"
    
    # scipy.io.savemat writes the dictionary directly to a MATLAB v5 MAT-file
    sio.savemat(mat_path, weights_dict)
    
    print("-" * 50)
    print(f"Success! MATLAB file saved to: {mat_path.resolve()}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Actor matrices to a .mat file")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the TD3 checkpoint (.zip file)",
    )
    parser.add_argument(
        "--output-dir",
        default="./extracted_weights",
        help="Directory to save the .mat file",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    extract_to_matlab(checkpoint_path, output_dir)

if __name__ == "__main__":
    main()