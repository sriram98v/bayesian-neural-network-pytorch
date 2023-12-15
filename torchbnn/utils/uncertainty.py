import torch

def epistemic_uncertainty(samples):
    return torch.var(samples, dim=1)

def aleatoric_uncertainty(samples):
    return -torch.sum(samples*torch.log(samples), dim=1)