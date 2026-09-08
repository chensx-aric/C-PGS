
from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = self.fc(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, feature_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, z):
        return self.fc(z)
class CasualHyperGraph(nn.Module):
    def __init__(self, feature_dim, latent_dim, num_classes, device='cuda'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.mask_generator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
        self.proto_weight = nn.Parameter(torch.tensor(1.0))
        self.encoder = Encoder(feature_dim, latent_dim)
        self.decoder = Decoder(latent_dim, feature_dim)
        # self.hgnn = nn.ModuleList([HGNN(feature_dim, feature_dim) for _ in range(5)])
        # 可学习的原型向量
        # self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))  # 初始化原型向量
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(latent_dim, num_classes)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.mask_projector = nn.Linear(feature_dim, feature_dim)
        # self.mask_projector = nn.Sequential(
        #     nn.Linear(feature_dim, feature_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(feature_dim, feature_dim)
        # )
    def perturb_with_mask(self, x, mask):

        B = x.size(0)
        ii = torch.randperm(B).to(x.device)
        feature_output1 = x[ii]
        lam = torch.rand(B, 1).to(x.device) * 0.5
        x_perturbed = mask * x + (1 - mask) * ((1 - lam) * x + lam * feature_output1)

        return x_perturbed
    def forward(self, x):

        # --- Step 2: 生成掩码 ---
        mask = self.mask_generator(x)  # [N, D]
        mask = self.mask_projector(mask)
        hard_mask = (mask > 0.5).float()  # hard mask

        # --- Step 3: 扰动非重要特征 ---
        x = self.perturb_with_mask(x, hard_mask)
        # --- Step 5: 编码器 ---
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # --- Step 6: 解码器 ---
        x_recon = self.decoder(z)
        # sim = F.cosine_similarity(x_recon.unsqueeze(1), self.prototypes.unsqueeze(0), dim=-1)
        logits = self.classifier(x_recon) #+ sim * self.proto_weight

        return logits, x_recon, mask