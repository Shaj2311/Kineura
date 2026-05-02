import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import VimeoDataset
from model.model import VideoEnhancementModel
import os

def train():
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Dataset and DataLoader
    dataset = VimeoDataset(root_dir='./data/vimeo_triplet', is_train=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Initialize Model
    model = VideoEnhancementModel().to(device)

    # Define Loss and Optimizer
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Configuration
    num_epochs = 5
    os.makedirs('checkpoints', exist_ok=True)

    # Training Loop
    model.train()
    for epoch in range(num_epochs):
        print(f"\n=== Epoch [{epoch+1}/{num_epochs}] ===")
        
        for batch_idx, (inputs, target) in enumerate(dataloader):
            inputs, target = inputs.to(device), target.to(device)

            # Forward pass
            output = model(inputs)
            loss = criterion(output, target)

            # Backward pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log progress
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} | Loss: {loss.item():.6f}")

        # Save epoch checkpoint
        checkpoint_path = f"checkpoints/kineura_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    print("Training pipeline complete!")

if __name__ == "__main__":
    train()