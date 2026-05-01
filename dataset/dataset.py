import random
import torchvision.transforms.functional as F
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class VimeoDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        # Define dataset root directory
        self.root_dir = root_dir

        # Load triplets
        list_file = 'tri_trainlist.txt' if is_train else 'tri_testlist.txt'
        self.list_path = os.path.join(root_dir, list_file)

        with open(self.list_path, 'r') as f:
            self.triplet_list = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.triplet_list)

    def __getitem__(self, idx):
        # Get folder path: e.g., '00001/0001'
        img_path = os.path.join(self.root_dir, 'sequences', self.triplet_list[idx])

        # Load three frames from triplet
        img1 = Image.open(os.path.join(img_path, 'im1.png'))
        img2 = Image.open(os.path.join(img_path, 'im2.png'))
        img3 = Image.open(os.path.join(img_path, 'im3.png'))

        # APPLY AUGMENTATION
        # Random Horizontal Flip
        if random.random() > 0.5:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)
            img3 = F.hflip(img3)

        # Random Vertical Flip
        if random.random() > 0.5:
            img1 = F.vflip(img1)
            img2 = F.vflip(img2)
            img3 = F.vflip(img3)

        # Convert to Tensor
        img1 = F.to_tensor(img1)
        img2 = F.to_tensor(img2)
        img3 = F.to_tensor(img3)

        # We return (input_frames, target_frame)
        # Input frames are usually concatenated or stacked
        inputs = torch.cat([img1, img2], dim=0) # Frame 1 and 2 are the input frames
        target = img3                            # Frame 3 is what we predict

        return inputs, target

