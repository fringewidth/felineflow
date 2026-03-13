import os
import torch

import matplotlib.pyplot as plt
from tqdm import tqdm
from modules import UNet, Diffusion, Config
import argparse

def generate_cats(model_path, num_samples=4, output_path="generated_cats.png"):
    device = Config.DEVICE
    
    # If path doesn't exist, check models directory
    if not os.path.exists(model_path):
        alt_path = os.path.join(Config.MODEL_DIR, model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
    
    print(f"Loading model from {model_path} on {device}...")

    
    model = UNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    diffusion = Diffusion(timesteps=Config.TIMESTEPS, device=device)
    
    print(f"Generating {num_samples} cats...")
    with torch.no_grad():
        # Start from pure noise
        x = torch.randn((num_samples, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)).to(device)
        
        # Iteratively denoise
        for i in tqdm(reversed(range(1, Config.TIMESTEPS)), desc="Denoising", total=Config.TIMESTEPS-1):
            t = (torch.ones(num_samples) * i).long().to(device)
            
            # Predict noise using model
            # We don't need autocast here as inference is less memory intensive, but it helps speed
            with torch.amp.autocast('mps', dtype=torch.float16):
                predicted_noise = model(x, t)
            
            alpha = diffusion.alpha[t][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            beta = diffusion.beta[t][:, None, None, None]
            
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            # DDPM Step
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
    
    # Post-process and save
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
    
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
    if num_samples == 1:
        axes = [axes]
    for i in range(num_samples):
        axes[i].imshow(x[i])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved generated images to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to unet_epoch_X.pth")
    parser.add_argument("--num", type=int, default=4, help="Number of cats to generate")
    args = parser.parse_args()
    
    generate_cats(args.weights, args.num)
