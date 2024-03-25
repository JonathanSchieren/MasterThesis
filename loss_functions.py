# imports 
import numpy as np
import torch
import torch.nn as nn

# Nash-Sutcliffe Efficiency

# NSE class
class NSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, obs):
        numerator = torch.sum((pred - obs)**2)
        denominator = torch.sum((obs - torch.mean(obs))**2)
        nse = torch.tensor([1.0], device=pred.device) - (numerator / denominator)
        return nse.item()

"""  
class NSELossMod(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, obs):
        numerator = torch.sum((pred - obs)**2)
        denominator = torch.sum((obs - torch.mean(obs))**2)
        nse = torch.tensor([1.0], device=pred.device) - (numerator / denominator)
        return -nse
"""

class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()

    def forward(self, pred, obs):
        epsilon = 1e-6  # Small constant to avoid division by zero
        log_pred = torch.log1p(torch.max(pred, torch.tensor(epsilon, device=pred.device)))  # Apply logarithm to predicted values
        log_obs = torch.log1p(torch.max(obs, torch.tensor(epsilon, device=obs.device)))  # Apply logarithm to target values
        squared_diff = (log_pred - log_obs) ** 2  # Calculate squared differences
        mean_squared_diff = torch.mean(squared_diff)  # Calculate mean squared difference
        rmsle = torch.sqrt(mean_squared_diff)  # Calculate RMSLE
        return rmsle
        
class MSELoss:
    def __call__(self, pred, obs):
        squared_error = (pred - obs) ** 2
        mse = torch.mean(squared_error)
        return mse.item()
        

# new and corrected
class KGELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, obs):
        # mean and std
        mean_pred = torch.mean(pred)
        mean_obs = torch.mean(obs)
        std_pred = torch.std(pred)
        std_obs = torch.std(obs)
        
        # number of items 
        n = obs.shape[0]

        # num and denom
        numerator = torch.sum((obs - mean_obs) * (pred - mean_pred))
        denom1 = torch.sum(torch.square(obs - mean_obs))
        denom2 = torch.sum(torch.square(pred - mean_pred))
        
        if denom2 == 0:
            kge = torch.tensor([-9999])
            return kge
        
        r = numerator / (torch.sqrt(denom1) * torch.sqrt(denom2))
        alpha = torch.sqrt(denom2/n) / torch.sqrt(denom1/n)
        beta = mean_pred / mean_obs
        
        kge = torch.tensor([1.0]) - torch.sqrt(torch.square(r - torch.tensor([1.0])) + torch.square(alpha - torch.tensor([1.0])) + torch.square(beta - torch.tensor([1.0])))
        
        return kge.item()
        
        
### is this correct? 
# new and corrected MODIFIED so it optimizes to 0 
class KGELossMOD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, obs):
        # mean and std
        mean_pred = torch.mean(pred)
        mean_obs = torch.mean(obs)
        std_pred = torch.std(pred)
        std_obs = torch.std(obs)
        
        # number of items 
        n = obs.shape[0]

        # num and denom
        numerator = torch.sum((obs - mean_obs) * (pred - mean_pred))
        denom1 = torch.sum(torch.square(obs - mean_obs))
        denom2 = torch.sum(torch.square(pred - mean_pred))
        
        if denom2 == 0:
            kge = torch.tensor([-9999])
            return kge
        
        r = numerator / (torch.sqrt(denom1) * torch.sqrt(denom2))
        alpha = torch.sqrt(denom2/n) / torch.sqrt(denom1/n)
        beta = mean_pred / mean_obs
        
        kge_mod = torch.sqrt(torch.square(r - torch.tensor([1.0])) + torch.square(alpha - torch.tensor([1.0])) + torch.square(beta - torch.tensor([1.0])))
        
        return kge_mod