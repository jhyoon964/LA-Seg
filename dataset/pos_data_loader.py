import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
# import odl
from torch_radon import RadonFanbeam
from skimage.transform import radon, iradon, resize

class posDataset(Dataset):
    def __init__(self):
        """
        Args:
            dataset_path (str): Path to the dataset directory.
            ids (list): List of folder names to load data from.
        """
                
    def radon_fanbeam(self, images, num_veiw=720):
        
        image_size = iamges.shape[-1]
        detector_count = 768
        source_distance = 600
        det_distance = 290
        angles = np.linspace(0, 2*np.pi, num_veiw, endpoint=False)
        
        radon = RadonFanbeam(
            image_size,
            det_count=detector_count,
            angles=angles,
            source_distance=source_distance,
            det_distance=det_distance,
        )
        
        sinogram = radon.forward(images)
        return sinogram        

    def __len__(self):
        return len(np.array([1]))

    def __getitem__(self):

        # Normalize the image data
        image = np.zeros((256, 512, 512))
        # Fill each channel with a unique 16x16 block set to 1
        for i in range(256):
            row_start = (i // 32) * 32  # Determine the row block position
            col_start = (i % 32) * 32   # Determine the column block position
            image[i, row_start:row_start+32, col_start:col_start+32] = 1
        
        sino = np.zeros((256, 720, 768))
        
        for i in range(256):
            sino[i] = self.radon_fanbeam(image[i])    
        
        
        full_sino = torch.tensor(sino)
        full_sino = (full_sino - torch.min(full_sino)) / (torch.max(full_sino) - torch.min(full_sino))
        
        
        return full_sino


class posDataLoader:
    def __init__(self, batch_size=1, shuffle=True, num_workers=0, train_val="train"):
        """
        Args:
            dataset (Dataset): Custom dataset class instance (AAPMDataset).
            batch_size (int): Number of samples per batch.
            shuffle (bool): If True, shuffle the data.
            num_workers (int): Number of subprocesses to use for data loading.
        """


    def radon_fanbeam(self, images, num_veiw=720):
    
        image_size = images.shape[-1]
        detector_count = 768
        source_distance = 600
        det_distance = 290
        angles = np.linspace(0, 2*np.pi, num_veiw, endpoint=False)
        
        radon = RadonFanbeam(
            image_size,
            det_count=detector_count,
            angles=angles,
            source_distance=source_distance,
            det_distance=det_distance,
        )
        images = torch.tensor(images, dtype=torch.float32).to('cuda')
        
        sinogram = radon.forward(images).to('cpu')
        return sinogram
    
    def get_loader(self):
        # Normalize the image data
        image = np.zeros((256, 512, 512))
        # Fill each channel with a unique 16x16 block set to 1
        for i in range(256):
            row_start = (i // 32) * 32  # Determine the row block position
            col_start = (i % 32) * 32   # Determine the column block position
            image[i, row_start:row_start+32, col_start:col_start+32] = 1
        
        sino = np.zeros((256, 720, 768))
        
        for i in range(256):
            sino[i] = self.radon_fanbeam(image[i])    
        
        
        full_sino = torch.tensor(sino).unsqueeze(0)
        full_sino = (full_sino - torch.min(full_sino)) / (torch.max(full_sino) - torch.min(full_sino))
        
        return full_sino
