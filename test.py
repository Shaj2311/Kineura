import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict

from dataset.dataset import VimeoDataset
from model.model import VideoEnhancementModel
from utils.metrics import Evaluator

def evaluateModel(weightsPath='checkpoints/best_weights.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f"Running on: {device} | GPUs: {num_gpus}")

    testDataset = VimeoDataset(root_dir='./data/vimeo_triplet', is_train=False)
    testLoader = DataLoader(testDataset, batch_size=32, shuffle=False)

    # Initialize the U-Net model and map it to the active hardware
    model = VideoEnhancementModel()

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Safely load pre-trained weights if they exist on the machine
    if os.path.exists(weightsPath):
        state_dict = torch.load(weightsPath, map_location=device)
        new_state_dict = OrderedDict()
        is_model_multi = isinstance(model, torch.nn.DataParallel)
        for k, v in state_dict.items():
            name = f'module.{k}' if (is_model_multi and not k.startswith('module.')) else (k[7:] if (not is_model_multi and k.startswith('module.')) else k)
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print(f"Loaded weights from {weightsPath}")

    evaluator = Evaluator(logFile="checkpoints/test_results.csv", device=device)
    model.eval()

    totalPsnr, totalSsim, totalLpips, totalLoss = 0, 0, 0, 0
    batchCount = 0

    print("\nStarting evaluation:")

    with torch.no_grad():
        for batchIdx, (inputs, target) in enumerate(testLoader):
            # Push batch data to GPU/CPU
            inputs, target = inputs.to(device), target.to(device)

            # Forward pass: Generate the predicted middle frame
            predictions = model(inputs)

            # Record metrics

            loss = F.mse_loss(predictions, target)
            totalLoss += loss.item()

            batchPsnr, batchSsim, batchLpips = evaluator.calculateBatchMetrics(target, predictions)

            totalPsnr += batchPsnr
            totalSsim += batchSsim
            totalLpips += batchLpips
            batchCount += 1

            print(f"Batch {batchIdx + 1}/{len(testLoader)} | Loss: {loss.item():.6f} | PSNR: {batchPsnr:.4f} | SSIM: {batchSsim:.4f}")

    if batchCount > 0:
        avgLoss = totalLoss / batchCount
        avgPsnr = totalPsnr / batchCount
        avgSsim = totalSsim / batchCount
        avgLpips = totalLpips / batchCount

        # Log to csv
        evaluator.logEpochData("Test", avgLoss, avgPsnr, avgSsim, avgLpips)

        print(f"\nFinal Evaluation:")
        print(f"Avg Loss: {avgLoss:.6f}")
        print(f"Avg PSNR: {avgPsnr:.4f}")
        print(f"Avg SSIM: {avgSsim:.4f}")
        print(f"Avg LPIPS: {avgLpips:.4f}")
        print(f"Results written to: {evaluator.logFile}")

if __name__ == "__main__":
    evaluateModel()
