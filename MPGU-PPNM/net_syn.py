import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== CBAM 注意力模块 ==========
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([max_out, avg_out], dim=1))
        x = x * sa
        return x


# ========== PPNM 网络 ==========
class PPNMModel(nn.Module):
    def __init__(self, band_dim, endmember_num, z_dim):
        super().__init__()
        self.P = endmember_num
        self.Channel = band_dim

        # ===== VAE Encoder =====
        self.fc1 = nn.Linear(band_dim, 32 * endmember_num)
        self.bn1 = nn.BatchNorm1d(32 * endmember_num)
        self.fc2 = nn.Linear(32 * endmember_num, 16 * endmember_num)
        self.bn2 = nn.BatchNorm1d(16 * endmember_num)
        self.fc3 = nn.Linear(16 * endmember_num, 4 * endmember_num)
        self.bn3 = nn.BatchNorm1d(4 * endmember_num)
        self.fc_mu = nn.Linear(4 * endmember_num, z_dim)
        self.fc_logvar = nn.Linear(4 * endmember_num, z_dim)

        # ===== VAE Decoder =====
        self.fc4 = nn.Linear(z_dim, endmember_num * 4)
        self.bn4 = nn.BatchNorm1d(endmember_num * 4)
        self.fc5 = nn.Linear(endmember_num * 4, endmember_num * 64)
        self.bn5 = nn.BatchNorm1d(endmember_num * 64)
        self.fc6 = nn.Linear(endmember_num * 64, band_dim * endmember_num)

        # ===== Nonlinear Parameter Decoder =====

        self.parameter_decoder = nn.Sequential(
            nn.Conv2d(band_dim, 128, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Tanh(), 
        )

        self.p = nn.Sequential(
            nn.Conv2d(band_dim*1, band_dim*1, kernel_size=1),
            nn.Softmax(dim=1),
        )


        # ===== Abundance Estimator with Pooling & Attention =====
        self.enc1 = nn.Sequential(
            nn.Conv2d(band_dim, 64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CBAMBlock(64)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            CBAMBlock(128)
        )

        self.up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.final_abundance = nn.Sequential(
            nn.Conv2d(64, endmember_num, kernel_size=1, padding=0),
            nn.BatchNorm2d(endmember_num),
            nn.Softmax(dim=1)
        )

    def encoder_z(self, x):
        h = F.leaky_relu(self.bn1(self.fc1(x)))
        h = F.leaky_relu(self.bn2(self.fc2(h)))
        h = F.leaky_relu(self.bn3(self.fc3(h)))
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z):
        h = F.leaky_relu(self.bn4(self.fc4(z)))
        h = F.leaky_relu(self.bn5(self.fc5(h)))
        em = torch.sigmoid(self.fc6(h))
        return em

    def estimate_abundance(self, x):
        B, _, H, W = x.shape
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x4 = self.up(x3)

        if x4.size(2) > H or x4.size(3) > W:
            x4 = x4[:, :, :H, :W]
        elif x4.size(2) < H or x4.size(3) < W:
            pad_h = H - x4.size(2)
            pad_w = W - x4.size(3)
            x4 = F.pad(x4, (0, pad_w, 0, pad_h))

        x5 = x4 + x1
        A = self.final_abundance(x5)
        return A

    def forward(self, x):  # x: [B, C, H, W]
        y = x.squeeze(0).view(x.shape[1], x.shape[2] * x.shape[3]).permute(1, 0)
        mu, log_var = self.encoder_z(y)
        z = self.reparameterize(mu, log_var)
        em = self.decoder(z)
        em_tensor = em.view([-1, self.P, self.Channel])
        E = torch.mean(em_tensor, dim=0).permute(1, 0).unsqueeze(0)  # [1, C, P]
        # E = End

        A = self.estimate_abundance(x)  # [B, P, H, W]
        B, P, H, W = A.shape
        A_flat = A.view(B, P, -1)

        linear = torch.bmm(E, A_flat)
        linear_img = linear.view(B, self.Channel, H, W)
        nonlinear = linear ** 2
        nonlinear_img = nonlinear.view(B, self.Channel, H, W)

        p_input = linear_img
        p_input = self.p(p_input)

        parameter = self.parameter_decoder(p_input)
        parameter= parameter.view(B, 1, -1)

        parameter = torch.clamp(parameter, min=-0.5, max=0.5) 
        

        recon = linear + parameter * nonlinear

        return em_tensor, E.squeeze(0), A.squeeze(0), linear_img, recon, mu, log_var, parameter
