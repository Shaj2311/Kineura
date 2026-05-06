import torch
import os
from torch.utils.data import DataLoader

from dataset.dataset import VimeoDataset
from model.model import VideoEnhancementModel
from utils.metrics import Evaluator

def evaluateModel(weightsPath='checkpoints/best_model.pth'):
    # Auto-detect GPU for fast inference, fallback to CPU locally
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # Load testing dataset (is_train=False is crucial here to load tri_testlist.txt)
    testDataset = VimeoDataset(root_dir='./data/vimeo_triplet', is_train=False)
    testLoader = DataLoader(testDataset, batch_size=4, shuffle=False)

    # Initialize the U-Net model and map it to the active hardware
    model = VideoEnhancementModel().to(device)
    
    # Safely load pre-trained weights if they exist on the machine
    if os.path.exists(weightsPath):
        model.load_state_dict(torch.load(weightsPath, map_location=device))
        print(f"Loaded weights from {weightsPath}")
    else:
        print(f"WARNING: No weights found at {weightsPath}. Testing with random initialization.")

    # Setup custom CSV logger for the final report metrics
    evaluator = Evaluator(logFile="final_test_results.csv")
    
    # Lock model layers (disables dropout/batchnorm updates during inference)
    model.eval() 
    
    totalPsnr = 0
    totalSsim = 0
    batchCount = 0

    print("\nStarting evaluation:")
    
    # Disable gradient tracking to save massive amounts of RAM and speed up testing
    with torch.no_grad(): 
        for batchIdx, (inputs, target) in enumerate(testLoader):
            # Push batch data to GPU/CPU
            inputs, target = inputs.to(device), target.to(device)

            # Forward pass: Generate the predicted middle frame
            predictions = model(inputs)
            
            # Calculate metrics for the current batch using your custom utils
            batchPsnr, batchSsim = evaluator.calculateBatchMetrics(target, predictions)
            
            totalPsnr += batchPsnr
            totalSsim += batchSsim
            batchCount += 1

            print(f"Batch {batchIdx + 1} | PSNR: {batchPsnr:.4f} | SSIM: {batchSsim:.4f}")

    # Compute and print the final average scores across all test batches
    if batchCount > 0:
        avgPsnr = totalPsnr / batchCount
        avgSsim = totalSsim / batchCount
        print(f"\nFinal Evaluation:")
        print(f"Average PSNR: {avgPsnr:.4f}")
        print(f"Average SSIM: {avgSsim:.4f}")

if __name__ == "__main__":
    evaluateModel()