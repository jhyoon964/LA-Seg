import torch
import numpy as np
from medpy import metric
import torch.nn as nn
def cal_psnr_torch(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def cal_psnr_np(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

class CAFL(nn.Module):
    def __init__(self):
        super(CAFL, self).__init__()

    def forward(self, caf, mask):
        feature_diff = (caf[0,:,:] - caf[1,:,:]) * mask

        loss = torch.sigmoid(-torch.norm(feature_diff, p=2, dim=(1, 2, 3))).mean()
        return loss

def gradient_loss(pred, target):

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)

    grad_pred_x = torch.nn.functional.conv2d(pred.unsqueeze(1), sobel_x, padding=1)
    grad_pred_y = torch.nn.functional.conv2d(pred.unsqueeze(1), sobel_y, padding=1)
    grad_target_x = torch.nn.functional.conv2d(target.unsqueeze(1), sobel_x, padding=1)
    grad_target_y = torch.nn.functional.conv2d(target.unsqueeze(1), sobel_y, padding=1)

    grad_diff_x = torch.abs(grad_pred_x - grad_target_x)
    grad_diff_y = torch.abs(grad_pred_y - grad_target_y)

    return torch.mean(grad_diff_x) + torch.mean(grad_diff_y)

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
