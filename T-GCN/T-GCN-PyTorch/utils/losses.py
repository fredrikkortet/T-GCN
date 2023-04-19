import torch
import math

def mse_with_regularizer_loss(inputs, targets, model, lamda=1.5e-3):
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) 
    reg_loss = lamda * reg_loss
    mse_loss = torch.sum((inputs - targets) ** 2) / (inputs.size(dim=0)*inputs.size(dim=1))
    return mse_loss + reg_loss

def mse_with_regularizer_l1_loss(inputs, targets, model, lamda=1.5e-3):
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(torch.abs(param))
    reg_loss = lamda * reg_loss
    mse_loss = torch.sum((inputs - targets) ** 2) / (inputs.size(dim=0)*inputs.size(dim=1))
    return mse_loss + reg_loss

def mse_with_regularizer_entropy_reg_loss(inputs, targets, model, lamda=1.5e-3):
    reg_loss = 0.0
    softmax = torch.nn.Softmax(dim=1)
    log_softmax = torch.nn.LogSoftmax(dim=1)
    reg_loss = torch.sum(-softmax(inputs)*log_softmax(inputs))
    reg_loss = lamda * reg_loss
    mse_loss = torch.sum((inputs - targets) ** 2) / (inputs.size(dim=0)*inputs.size(dim=1))
    return mse_loss + reg_loss

def mse_with_regularizer_log_cosh_loss(inputs, targets):
    reg_loss = 0.0
    reg_loss = torch.sum(torch.log(torch.cosh((inputs-targets)))) / 2
    reg_loss = reg_loss
    return reg_loss

def mse_with_regularizer_entropy_loss(inputs, targets, model, sigma=0.1):
    reg_loss = 0.0
    reg_loss += (1/(2*sigma**2))*torch.sum(torch.abs(targets-inputs))**2 + math.log(sigma)
    mse_loss = torch.sum((inputs - targets) ** 2) / (inputs.size(dim=0)*inputs.size(dim=1))
    return mse_loss + reg_loss
