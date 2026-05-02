import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    # doing two convs back to back is repeated everywhere so making a reusable block
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convBlock(x)

class VideoEnhancementModel(nn.Module):
    def __init__(self):
        super(VideoEnhancementModel, self).__init__()

        # input layer: takes our 6 stacked channels
        self.inputConv = DoubleConv(6, 64)
        
        # encoder: shrinks image and extracts patterns
        self.downBlock1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.downBlock2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # bottleneck: most compressed state
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # decoder: scales back up and merges with skip connections
        self.upScale1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.processUp1 = DoubleConv(512, 256) # 512 because we concatenate 256+256 from the skip

        self.upScale2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.processUp2 = DoubleConv(256, 128) 

        self.upScale3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.processUp3 = DoubleConv(128, 64)  

        # final output layer squashes down to 3 RGB channels for the predicted frame
        self.outputLayer = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, inputTensor):
        # going down
        features1 = self.inputConv(inputTensor)     
        features2 = self.downBlock1(features1)  
        features3 = self.downBlock2(features2)  
        
        compressedData = self.bottleneck(features3) 

        # coming back up + skip connections
        up1 = self.upScale1(compressedData)
        merged1 = torch.cat([features3, up1], dim=1) # copy paste high res data
        out1 = self.processUp1(merged1)

        up2 = self.upScale2(out1)
        merged2 = torch.cat([features2, up2], dim=1)
        out2 = self.processUp2(merged2)

        up3 = self.upScale3(out2)
        merged3 = torch.cat([features1, up3], dim=1)
        out3 = self.processUp3(merged3)

        finalPrediction = self.outputLayer(out3)
        
        return finalPrediction

# quick test to make sure shapes match up
if __name__ == '__main__':
    dummyInput = torch.randn(4, 6, 256, 256)
    testModel = VideoEnhancementModel()
    dummyOutput = testModel(dummyInput)
    
    print("unet works!")
    print(f"in:  {dummyInput.shape}")
    print(f"out: {dummyOutput.shape}")