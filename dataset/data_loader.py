import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
from torch_radon import RadonFanbeam
from skimage.transform import radon, iradon, resize
from PIL import Image
import random
from scipy.ndimage import rotate
class AAPMDataset(Dataset):
    def __init__(self, dataset_path, ids, bank_path, augment=True):
        """
        Args:
            dataset_path (str): Path to the dataset directory.
            ids (list): List of folder names to load data from.
        """
        self.dataset_path = dataset_path
        self.augment = augment
        
        self.data_file_paths = []
        for folder_id in ids:
            folder_path = os.path.join(self.dataset_path, folder_id)
            self.data_file_paths.extend(glob(os.path.join(folder_path, '*.npy')))
        # for folder_id in ids:
        #     folder_path = os.path.join(self.dataset_path, folder_id)
        #     self.data_file_paths.extend(glob(os.path.join(folder_path, '*.png')))
        self.pos_data = np.load(bank_path)
        
        
    def radon_parallel(self, images, num_view=720, start_ang=0, end_ang=360):
        """
        Args:
            images (numpy.ndarray): Input images with shape (ch, h, w).
            num_view (int): Number of projection angles.
            start_ang (float): Start angle in radians.
            end_ang (float): End angle in radians.
            num_detectors (int): Number of detector bins.
        
        Returns:
            numpy.ndarray: Sinogram with shape (ch, num_detectors, num_view).
        """
        
        channels, h, w = images.shape
        sinograms = []
        thetas = np.linspace(start_ang, end_ang, num_view, endpoint=False)
        
        # Process each channel separately
        for c in range(channels):
            # Apply Radon transform on each individual channel
            sinogram = radon(resize(images[c], (512, 512), anti_aliasing=True), theta=thetas, circle=False)
            sinograms.append(sinogram)
        
        # Stack all channel sinograms along the first dimension
        sinograms = np.stack(sinograms)
        
        return sinograms  # Shape (ch, num_detectors, num_view)
        
    def apply_augmentation(self, image, label):
        """
        Apply augmentation (rotation, flip, etc.) to the image.
        Args:
            image (numpy.ndarray): Input image with shape (1, h, w).
        Returns:
            numpy.ndarray: Augmented image.
        """
        # Random rotation
        if random.random() < 0.5:
            # angle = random.choice([90, 180, 270])
            angle = random.uniform(-45, 45)  # Rotate within ±30 degrees
            image = rotate(image.squeeze(0), angle=angle, order=0, reshape=False)
            label = rotate(label.squeeze(0), angle=angle, order=0, reshape=False)
            # image = np.rot90(image.squeeze(0), k=angle // 90)#.copy()  # Ensure no negative stride by copying the array
            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)
        # Random horizontal flip
        if random.random() < 0.5:
            image = np.flip(image, axis=2)#.copy()  # Ensure no negative stride by copying the array
            label = np.flip(label, axis=2)
            
        return image, label
    
    def radon_fanbeam(self, images, num_view=720):
        
        image_size = images.shape[-1]
        detector_count = 768
        source_distance = 600
        det_distance = 290
        angles = np.linspace(0, 2*np.pi, num_view, endpoint=False)
        radon = RadonFanbeam(
            image_size,
            det_count=detector_count,
            angles=angles,
            source_distance=source_distance,
            det_distance=det_distance,
        )
        
        images = torch.FloatTensor(images).to('cuda')
        sinogram = radon.forward(images)[0]

        return sinogram
        
    def __len__(self):
        return len(self.data_file_paths)

    def __getitem__(self, idx):
        # Load the .npy file
        img_path = self.data_file_paths[idx]
        label_path = img_path.replace("image", "label")
        # image = np.load(img_path)
        file_name = os.path.basename(img_path)
        image = np.array(np.load(img_path))  # Convert to grayscale
        label = np.array(np.load(label_path))
        # image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        # label = np.expand_dims(label, axis=0)
        
        # Normalize the image data
        # image = resize(image, (1, 512, 512), anti_aliasing=True)
        
        # Apply data augmentation if enabled
        if self.augment:
            image, label = self.apply_augmentation(image, label)
        if np.max(image) != np.min(image):
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # label = label / 255.
        label = np.where(label > 0, 1, 0)
        # print(image.shape)
        
        sinogram = self.radon_fanbeam(image)
        sino_label = self.radon_fanbeam(label)        
        # Convert to tensor
        image_tensor = torch.tensor(image, dtype=torch.float32)
        
        full_sino = torch.tensor(sinogram)
        max_value = torch.max(full_sino)
        min_value = torch.min(full_sino)
        full_sino = (full_sino - torch.min(full_sino)) / (torch.max(full_sino) - torch.min(full_sino))
        
        sino_label = torch.tensor(sino_label)
        
        return image_tensor[0], full_sino, self.pos_data, max_value, min_value, sino_label, label, file_name


class AAPMDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0, train_val="train", augment=True):
        """
        Args:
            dataset (Dataset): Custom dataset class instance (AAPMDataset).
            batch_size (int): Number of samples per batch.
            shuffle (bool): If True, shuffle the data.
            num_workers (int): Number of subprocesses to use for data loading.
        """
        if dataset.dataset_name == "AAPM":
            self.dataset_path = dataset.dataset_path
            if train_val == "train":
                self.ids = dataset.train_set
                # Initialize the dataset and DataLoader
                self.dataset = AAPMDataset(self.dataset_path, self.ids, bank_path="./aux_sino_seg/pos_data/sino_bank.npy",augment=False)
            else:
                self.ids = dataset.val_set
                # Initialize the dataset and DataLoader
                self.dataset = AAPMDataset(self.dataset_path, self.ids, bank_path="./aux_sino_seg/pos_data/sino_bank.npy",augment=False)

        
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_loader(self):
        return self.dataloader
