import torch 

def smooth_L1_loss(deltas, targets, sigma=3.0):
    sigma2 = sigma**2 
    diffs = deltas - targets 
    smooth_l1_signs = torch.lt(torch.abs(diffs), 1.0 / sigma2).float()

    smooth_L1_option1 = torch.mul(diffs, diffs) * 0.5 * sigma2 
    smooth_L1_option2 = torch.abs(diffs) - 0.5 / sigma2
    smooth_L1_add = torch.mul(smooth_L1_option1, smooth_L1_option2) +\
         torch.mul(smooth_L1_option2, 1 - smooth_l1_signs)

    return smooth_L1_add