import torch
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnrMetric
from skimage.metrics import structural_similarity as ssimMetric
import lpips 

class Evaluator:
    def __init__(self, logFile="metrics_log.csv", device=None):
        self.logFile = logFile
        
        # Auto-detect device so LPIPS runs on the GPU for speed
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the LPIPS model (VGG network) and push to device
        self.lpips_metric = lpips.LPIPS(net='vgg').to(self.device)

        # setup csv for final report (Added lpips to header)
        if not os.path.exists(self.logFile):
            with open(self.logFile, "w") as f:
                f.write("epoch,loss,psnr,ssim,lpips\n")

    def calculateBatchMetrics(self, groundTruth, prediction):
        batchPsnr = 0
        batchSsim = 0
        batchSize = groundTruth.shape[0]

        # drop to cpu for skimage and clamp to [0, 1] to avoid range errors
        gtArray = torch.clamp(groundTruth, 0, 1).detach().cpu().numpy()
        predArray = torch.clamp(prediction, 0, 1).detach().cpu().numpy()

        for i in range(batchSize):
            # fix shape to [H, W, C]
            gtImg = gtArray[i].transpose(1, 2, 0)
            predImg = predArray[i].transpose(1, 2, 0)

            pVal = psnrMetric(gtImg, predImg, data_range=1.0)
            sVal = ssimMetric(gtImg, predImg, data_range=1.0, channel_axis=2) # axis 2 is rgb

            batchPsnr += pVal
            batchSsim += sVal

        #  LPIPS Calculation 
        # Convert tensors from [0, 1] range to [-1, 1] for LPIPS
        gt_lpips = (groundTruth * 2.0) - 1.0
        pred_lpips = (prediction * 2.0) - 1.0
        
        # Ensure they are on the correct device
        gt_lpips = gt_lpips.to(self.device)
        pred_lpips = pred_lpips.to(self.device)

        # Calculate LPIPS for the whole batch at once and get the average float
        batchLpips = self.lpips_metric(pred_lpips, gt_lpips).mean().item()

        return batchPsnr / batchSize, batchSsim / batchSize, batchLpips

    def logEpochData(self, epoch, avgLoss, avgPsnr, avgSsim, avgLpips):
        with open(self.logFile, "a") as f:
            f.write(f"{epoch},{avgLoss:.6f},{avgPsnr:.4f},{avgSsim:.4f},{avgLpips:.4f}\n")