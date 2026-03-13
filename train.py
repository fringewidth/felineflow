import os
import re
import zipfile
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Config, Diffusion, UNet

def setup_dirs():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.SAMPLE_DIR, exist_ok=True)

def find_latest_checkpoint():
    checkpoints = []
    # also check project root for backwards compatibility
    for f in os.listdir("."):
        if f.startswith("unet_epoch_") and f.endswith(".pth"):
            match = re.search(r"unet_epoch_(\d+).pth", f)
            if match:
                checkpoints.append((".", f, int(match.group(1))))
    
    if os.path.exists(Config.MODEL_DIR):
        for f in os.listdir(Config.MODEL_DIR):
            if f.startswith("unet_epoch_") and f.endswith(".pth"):
                match = re.search(r"unet_epoch_(\d+).pth", f)
                if match:
                    checkpoints.append((Config.MODEL_DIR, f, int(match.group(1))))
    
    if not checkpoints:
        return None, 0
    
    checkpoints.sort(key=lambda x: x[2], reverse=True)
    latest = checkpoints[0]
    return os.path.join(latest[0], latest[1]), latest[2]

def setup_dataset():
    data_ready = False
    if os.path.exists(Config.DATASET_PATH):
        image_count = sum([len(files) for r, d, files in os.walk(Config.DATASET_PATH) if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)])
        if image_count > 0:
            data_ready = True
            print(f"Dataset already exists with {image_count} images.")
    
    if not data_ready:
        print("Dataset not found or empty. Setting up...")
        os.makedirs(Config.DATASET_PATH, exist_ok=True)
        
        is_pointer = False
        if os.path.exists(Config.ZIP_PATH):
            if os.path.getsize(Config.ZIP_PATH) < 1000:
                is_pointer = True
                print("Found Git LFS pointer for cats_processed.zip. Need to download real file.")
        
        if not os.path.exists(Config.ZIP_PATH) or is_pointer:
            print(f"Downloading dataset from {Config.ZIP_URL}...")
            response = requests.get(Config.ZIP_URL, stream=True)
            with open(Config.ZIP_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print("Extracting dataset...")
        try:
            with zipfile.ZipFile(Config.ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(Config.DATASET_PATH)
            print("Dataset extracted.")
            
            root_files = [f for f in os.listdir(Config.DATASET_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if root_files:
                os.makedirs(os.path.join(Config.DATASET_PATH, "cats"), exist_ok=True)
                for f in root_files:
                    os.rename(os.path.join(Config.DATASET_PATH, f), os.path.join(Config.DATASET_PATH, "cats", f))
                print(f"Moved {len(root_files)} images to dataset/cats for ImageFolder compatibility.")
                
        except zipfile.BadZipFile:
            print("Error: The downloaded file is not a valid zip. Deleting it.")
            if os.path.exists(Config.ZIP_PATH): os.remove(Config.ZIP_PATH)
            return

def save_samples(model, diffusion, epoch):
    model.eval()
    with torch.no_grad():
        n = 4
        x = torch.randn((n, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)).to(Config.DEVICE)
        for i in tqdm(reversed(range(1, Config.TIMESTEPS)), desc="Sampling", position=0, leave=False):
            t = (torch.ones(n) * i).long().to(Config.DEVICE)
            with torch.amp.autocast('mps', dtype=torch.float16):
                predicted_noise = model(x, t)
            alpha = diffusion.alpha[t][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            beta = diffusion.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
    
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
    
    fig, axes = plt.subplots(1, n, figsize=(12, 3))
    for i in range(n):
        axes[i].imshow(x[i])
        axes[i].axis('off')
    
    save_path = os.path.join(Config.SAMPLE_DIR, f"sample_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()
    model.train()

def train():
    if Config.DEVICE == "mps":
        torch.mps.set_per_process_memory_fraction(Config.GPU_MEMORY_RATIO)
    setup_dirs()
    setup_dataset()
    
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE + 32, Config.IMAGE_SIZE + 32)),
        transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.ImageFolder(root=Config.DATASET_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

    model = UNet().to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    mse = nn.MSELoss()
    diffusion = Diffusion(timesteps=Config.TIMESTEPS, device=Config.DEVICE)
    scaler = torch.amp.GradScaler('mps')

    # resume logic
    ckpt_path, start_epoch = find_latest_checkpoint()
    if ckpt_path:
        print(f"Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=Config.DEVICE)
        
        # Determine if this is a new-style dict or old-style weights-only
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Full state loaded. Resuming from epoch {start_epoch}")
        else:
            # Fallback for older .pth files that only contain state_dict
            model.load_state_dict(checkpoint)
            # Adjust scheduler for resume if we only have weights
            for _ in range(start_epoch + 1):
                scheduler.step()
            start_epoch += 1
            print(f"Weights-only loaded. Estimated resume from epoch {start_epoch}")
    else:
        print("Starting training from scratch.")
        start_epoch = 0

    print(f"Starting training on {Config.DEVICE}...")
    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        pbar = tqdm(dataloader)
        epoch_loss = []
        for i, (images, _) in enumerate(pbar):
            images = images.to(Config.DEVICE)
            t = diffusion.sample_timesteps(images.shape[0]).to(Config.DEVICE)
            x_noisy, noise = diffusion.noise_images(images, t)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('mps', dtype=torch.float16):
                predicted_noise = model(x_noisy, t)
                loss = mse(noise, predicted_noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss.append(loss.item())
            pbar.set_description(f"Epoch {epoch} Loss: {np.mean(epoch_loss):.4f} LR: {scheduler.get_last_lr()[0]:.2e}")
        
        scheduler.step()
        
        save_samples(model, diffusion, epoch)
        save_path = os.path.join(Config.MODEL_DIR, f"unet_epoch_{epoch}.pth")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }
        torch.save(checkpoint, save_path)
        print(f"Saved full checkpoint and samples for epoch {epoch}")

if __name__ == "__main__":
    train()
