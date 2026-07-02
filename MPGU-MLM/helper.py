import scipy.io as sio
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.optimize import nnls
from scipy.optimize import linear_sum_assignment
def load_data(file_path):
    data = sio.loadmat(file_path)
    Y = torch.from_numpy(data["Y"]).float()
    Abu = torch.from_numpy(data["A"]).float()
    M = torch.from_numpy(data["M"]).float()
    Y = torch.reshape(Y, (Y.shape[0], int(np.sqrt(Y.shape[1])), int(np.sqrt(Y.shape[1]))))
    Abu = torch.reshape(Abu, (Abu.shape[0], int(np.sqrt(Abu.shape[1])), int(np.sqrt(Abu.shape[1]))))
    print(f"数据形状 - 光谱: {Y.shape}, 丰度: {Abu.shape}")

    return Y, Abu, M

def reconstruction_SADloss(output, target):
    dot_product = (output * target).sum(dim=0)
    output_norm = torch.norm(output, dim=0)
    target_norm = torch.norm(target, dim=0)

    eps = 1e-6
    denom = output_norm * target_norm + eps
    cosine_sim = dot_product / denom
    cosine_clamped = torch.clamp(cosine_sim, -1.0 + 1e-6, 1.0 - 1e-6)  # 更安全
    theta = torch.acos(cosine_clamped)

    if torch.isnan(theta).any():
        print("SAD contains NaN")

    return theta.mean()


# abundance normalization
def norm_abundance_GT(abundance_input, abundance_GT_input, endmember_number, Nx,Ny):
    abundance_input = abundance_input / (torch.sum(abundance_input, dim=0))
    abundance_input = torch.reshape(
        abundance_input, (endmember_number, Nx,Ny)
    )
    abundance_input = abundance_input.cpu().detach().numpy()
    abundance_GT_input = abundance_GT_input / (torch.sum(abundance_GT_input, dim=0))
    abundance_GT_input = abundance_GT_input.cpu().detach().numpy()
    return abundance_input, abundance_GT_input

# endmember normalization
def norm_endmember(endmember_input, endmember_GT, endmember_number):
    for i in range(0, endmember_number):
        endmember_input[:, i] = endmember_input[:, i] / np.max(endmember_input[:, i])
        endmember_GT[:, i] = endmember_GT[:, i] / np.max(endmember_GT[:, i])
    return endmember_input, endmember_GT

# calculate RMSE of abundance
def AbundanceRmse(inputsrc, inputref):
    rmse = np.sqrt(((inputsrc - inputref) ** 2).mean())
    return rmse

# calculate SAD of endmember
def SAD_distance(src, ref):
    cos_sim = np.dot(src, ref) / ((np.linalg.norm(src)+1e-6) * (np.linalg.norm(ref)+1e-6))
    SAD_sim = np.arccos(cos_sim)
    return SAD_sim


# change the index of abundance and endmember
def arange_A_E(abundance_input, abundance_GT_input, endmember_input, endmember_GT, endmember_number):
    RMSE_matrix = np.zeros((endmember_number, endmember_number))
    SAD_matrix = np.zeros((endmember_number, endmember_number))
    RMSE_index = np.zeros(endmember_number).astype(int)
    SAD_index = np.zeros(endmember_number).astype(int)
    RMSE_abundance = np.zeros(endmember_number)
    SAD_endmember = np.zeros(endmember_number)

    for i in range(0, endmember_number):
        for j in range(0, endmember_number):
            RMSE_matrix[i, j] = AbundanceRmse(
                abundance_input[i, :, :], abundance_GT_input[j, :, :]
            )
            SAD_matrix[i, j] = SAD_distance(endmember_input[:, i], endmember_GT[:, j])

        RMSE_index[i] = np.argmin(RMSE_matrix[i, :])
        SAD_index[i] = np.argmin(SAD_matrix[i, :])
        RMSE_abundance[i] = np.min(RMSE_matrix[i, :])
        SAD_endmember[i] = np.min(SAD_matrix[i, :])

    abundance_input[np.arange(endmember_number), :, :] = abundance_input[
        RMSE_index, :, :
    ]
    endmember_input[:, np.arange(endmember_number)] = endmember_input[:, SAD_index]


    return abundance_input, abundance_GT_input, endmember_input, endmember_GT,RMSE_abundance, SAD_endmember




