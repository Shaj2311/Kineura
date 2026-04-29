import torch
import torch.nn as nn

class VideoEnhancementModel(nn.Module):
    def __init__(self):
        super(VideoEnhancementModel, self).__init__()

        # Layer 1: Expands from 6 channels (2 frames) to 16
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        # Layer 2: Keeps features at 16 channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # Layer 3: Compresses 16 channels down to 3 channels (1 final frame)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        modelOutput = self.conv3(x) 

        return modelOutput

# --- Testing the Model (Runs only if this file is executed directly) ---
if __name__ == "__main__":
    # Simulates a batch of 4 inputs, 6 channels (2 frames), 256x256 resolution
    dummyInput = torch.randn(4, 6, 256, 256)

    enhancementModel = VideoEnhancementModel()
    dummyOutput = enhancementModel(dummyInput)

    print("Test successful!")
    print(f"Input shape fed to model:  {dummyInput.shape}")
    print(f"Output shape from model: {dummyOutput.shape}")

