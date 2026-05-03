import torch
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnrMetric
from skimage.metrics import structural_similarity as ssimMetric

class Evaluator:
    def __init__(self, logFile="metrics_log.csv"):
        self.logFile = logFile

        # setup csv for final report
        if not os.path.exists(self.logFile):
            with open(self.logFile, "w") as f:
                f.write("epoch,loss,psnr,ssim\n")

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

        return batchPsnr / batchSize, batchSsim / batchSize

    def logEpochData(self, epoch, avgLoss, avgPsnr, avgSsim):
        with open(self.logFile, "a") as f:
            f.write(f"{epoch},{avgLoss:.6f},{avgPsnr:.4f},{avgSsim:.4f}\n")
