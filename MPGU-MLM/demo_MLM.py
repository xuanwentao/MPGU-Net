import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import random
from helper import *
from net_syn import *
from sklearn.cluster import KMeans
import time
Stime = time.time()
seed = 1234
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
selected_dataset = "MLMdata50"
run_all = False  # True: 跑所有数据集，False: 跑 selected_dataset

dataset_config = {
    "MLMdata50": {
        "data_path": "MLMdata50.mat",
        "save_path": "./result/resultourMLM_MLMdata50.mat",
        "lamba_re": 1,
        "lamba_lin_non": 1,
        "lamba_lin": 0,
        "lamba_vol":1e-6,
        "lamba_sad":1e-2,
        "lamba_kl": 1e-1,
        "epochs": 10,
        "z_dim": 2,
        "lr":0.001,
        "step_size":100,
        "gamma":0.9
    },
}

class HSDataset(Dataset):
    def __init__(self, Y):
        self.Y = Y

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.Y
    
def train(model, dataloader, epochs, lamba_re,lamba_lin_non,lamba_lin, lamba_vol, lamba_sad, lamba_kl):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    MSE = torch.nn.MSELoss(size_average=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    for epoch in range(epochs):
        total_loss = 0
        for x1 in dataloader:
            scheduler.step()
            x1 = x1.to(device)
            optimizer.zero_grad()
            em_tensor, em, abundance, re_linear, recon, mu, log_var, parameter = model(x1)

            recon = torch.reshape(recon, (x1.shape[1], -1))
            re_linear = torch.reshape(re_linear, (x1.shape[1], -1))
            x1 = torch.reshape(x1, (x1.shape[1], -1))
            safe_para = parameter.squeeze(0)
            denom = 1 - safe_para+ safe_para * x1
            denom = denom + 1e-6  
            linear_cap = x1 / denom

            m = linear_cap.mean(dim=1, keepdim=True)
            one_r_T = torch.ones((1, em.shape[1]), dtype=em.dtype, device=em.device)
            m1T = m@one_r_T

            loss_re =reconstruction_SADloss(x1,recon)
            loss_relinear_non = reconstruction_SADloss(re_linear,linear_cap)
            loss_relinear = reconstruction_SADloss(x1,re_linear)
            loss_vol = torch.norm(em - m1T, p='fro') ** 2

            m_repeat = m.repeat(1, em.shape[1])  # shape: [156, 3]
            loss_sad = reconstruction_SADloss(em, m_repeat)

            

            kl_div = -0.5 * (log_var + 1 - mu ** 2 - log_var.exp())
            kl_div = kl_div.sum() / em_tensor.shape[0]
            kl_div = torch.max(kl_div, torch.tensor(0.2).to(device))


            loss = lamba_re* loss_re + lamba_lin_non * loss_relinear_non + lamba_lin * loss_relinear + lamba_vol*loss_vol + lamba_sad*loss_sad+lamba_kl * kl_div


            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, total_loss: {total_loss/len(dataloader):.4f}, "
              f"loss_re: {loss_re.item():.4f}, "
              f"loss_relinear_non: {loss_relinear_non.item():.4f}, "
              f"loss_relinear: {loss_relinear.item():.4f}, "
              f"loss_vol: {loss_vol.item():.4f}, "
              f"kl_div: {kl_div:.4f}")

def evaluate_and_save(model, Y, Abu, End, config):
    model.eval()
    with torch.no_grad():
        Y = Y.unsqueeze(0).to(device)
        em_tensor, em, abundance, re_linear, recon, mu, log_var, parameters = model(Y)

        EndNum, Nx,Ny = Abu.shape[0], Y.shape[2],Y.shape[3]
        abundance = torch.reshape(abundance, (EndNum, Nx,Ny))

        em = em.detach().cpu().numpy()
        End = End.detach().cpu().numpy()
        re = recon.squeeze(0).detach().cpu().numpy()
        parameters = parameters.squeeze(0).detach().cpu().numpy()

        abundance, Abu = norm_abundance_GT(abundance, Abu, EndNum, Nx,Ny)
        em, End = norm_endmember(em, End, EndNum)

        abundance_input, abundance_GT_input, endmember_input, endmember_GT,RMSE_abundance, SAD_endmember = arange_A_E(
            abundance, Abu, em, End, EndNum)

        print("RMSE", RMSE_abundance)
        print("mean_RMSE", RMSE_abundance.mean())
        print("endmember_SAD", SAD_endmember)
        print("mean_SAD", SAD_endmember.mean())


        abu = np.reshape(abundance_input, (EndNum, Nx* Ny))
        em_tensor = em_tensor.detach().cpu().numpy()

        os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
        Etime = time.time()
        Time = [Etime-Stime]
        sio.savemat(config["save_path"], {'Aest': endmember_input, 'Sest': abu, 're': re,'em_tensor':em_tensor,'Time':Time,'parameters':parameters})

if __name__ == "__main__":
    target_datasets = dataset_config.keys() if run_all else [selected_dataset]

    for dataset_name in target_datasets:
        print(f"\n=== Processing {dataset_name} ===")
        config = dataset_config[dataset_name]

        Y, Abu, End = load_data(config["data_path"])
        Nb, col = Y.shape[0], Y.shape[1]
        EndNum = Abu.shape[0]

        dataset = HSDataset(Y)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        model = MLMModel(Nb, EndNum,config["z_dim"]).to(device)

        train(model, dataloader, epochs=config["epochs"],
              lamba_re=config["lamba_re"],
              lamba_lin_non=config["lamba_lin_non"],
              lamba_lin=config["lamba_lin"],
              lamba_vol=config["lamba_vol"],
              lamba_sad=config["lamba_sad"],
              lamba_kl=config["lamba_kl"])

        evaluate_and_save(model, Y, Abu, End, config)
