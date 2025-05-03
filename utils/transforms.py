from skimage.transform import radon, iradon, resize
import numpy as np
from scipy.interpolate import interp1d
from torch_radon import RadonFanbeam
import torch
# import odl


def iradon_parallel(sinograms, num_view=720, start_ang=0, end_ang=360, output_size=(512, 512)):
    """
    Args:
        sinograms (numpy.ndarray): Sinograms with shape (ch, num_detectors, num_view).
        num_view (int): Number of projection angles.
        start_ang (float): Start angle in radians.
        end_ang (float): End angle in radians.
        num_detectors (int): Number of detector bins.
        output_size (tuple): Size of the output image.
    
    Returns:
        numpy.ndarray: Reconstructed images with shape (ch, h, w).
    """
    
    _, _ = sinograms.shape
    images = []
    thetas = np.linspace(start_ang, end_ang, num_view, endpoint=False)
    
    image = iradon(sinograms, theta=thetas, circle=False)
    images.append(image)

    images = np.stack(images)
    
    return images 

def interpolate_1d(sino_img, theta):
    
    sampled_angles = theta[::8]
    sv_sino = sino_img[:, ::8]

    interpolated_sinogram = np.zeros_like(sino_img)
    x_sampled = np.arange(0, sv_sino.shape[1]) * 8
    x_full = np.arange(sino_img.shape[1])

    for i in range(sino_img.shape[0]):
        interp_func = interp1d(x_sampled, sv_sino[i, :], kind='linear', fill_value="extrapolate")
        interpolated_sinogram[i, :] = interp_func(x_full)
    return interpolated_sinogram


def radon_fanbeam(sinogram, num_view=720):
    image_size = 512 
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
    filtered_sinogram = radon.filter_sinogram(sinogram, "ram-lak")
    reconstructed_image = radon.backward(filtered_sinogram)
    
    return reconstructed_image    
    
    
    