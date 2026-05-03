import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import VimeoDataset
from model.model import VideoEnhancementModel
from utils.metrics import Evaluator
import os

def train():
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Dataset and DataLoader
    dataset = VimeoDataset(root_dir='./data/vimeo_triplet', is_train=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=16)

    # Initialize Model
    #model = VideoEnhancementModel().to(device)
    model = VideoEnhancementModel()
    evaluator = Evaluator(logFile="checkpoints/metrics_log.csv")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Define Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Configuration
    num_epochs = 5

    # Automatically creating  folder if don  exist
    os.makedirs('checkpoints', exist_ok=True)

    #  WEIGHT PERSISTENCE & GLOBAL BEST SETUP


    # Save starting weights
    #torch.save(model.state_dict(), 'checkpoints/starting_weights.pth')
    state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_to_save, 'checkpoints/starting_weights.pth')
    print("Saved checkpoints/starting_weights.pth")

    # Check for existing best loss
    loss_file_path = 'checkpoints/best_loss.txt'
    best_loss = float('inf') # Default to infinity if  we've never run this before

    if os.path.exists(loss_file_path):
        with open(loss_file_path, 'r') as f:
            best_loss = float(f.read().strip())
            print(f" Loaded all-time best loss from previous runs: {best_loss:.6f}")

    # Training Loop
    model.train()
    for epoch in range(num_epochs):
        print(f"\n=== Epoch [{epoch+1}/{num_epochs}] ===")

        # Track total loss to calculate the epoch average later
        running_loss = 0.0

        for batch_idx, (inputs, target) in enumerate(dataloader):
            inputs, target = inputs.to(device), target.to(device)

            # Forward pass
            output = model(inputs)
            loss = criterion(output, target)

            # Backward pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add this batch's loss to running total
            running_loss += loss.item()

            # Log progress
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.6f}")

        #  END OF EPOCH CHECKPOINTING
        # Calculate the average loss across the whole epoch
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss:.6f}")

        # Calculate metrics for the last batch of the epoch
        avg_psnr, avg_ssim = evaluator.calculateBatchMetrics(target, output)
        evaluator.logEpochData(epoch + 1, epoch_loss, avg_psnr, avg_ssim)

        # Save latest weights (for  crash recovery)
        #torch.save(model.state_dict(), 'checkpoints/latest_weights.pth')
        state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state_to_save, 'checkpoints/latest_weights.pth')
        print("Saved checkpoints/latest_weights.pth")

        # Save best weights (ONLY if it beats all-time high score)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_to_save, 'checkpoints/best_weights.pth')

            with open(loss_file_path, 'w') as f:
                f.write(str(best_loss))

            print(f"🎉 NEW ALL-TIME BEST! Loss dropped to {best_loss:.6f}. Saved best_weights.pth")
        else:
            print(f"Loss ({epoch_loss:.6f}) did not beat the all-time best ({best_loss:.6f}).")

    print("Training pipeline complete!")

if __name__ == "__main__":
    train()
