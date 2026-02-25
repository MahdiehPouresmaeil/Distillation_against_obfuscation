import torch
import torch.nn as nn

from util.util import Random


class DeepSigns(nn.Module):
    def __init__(self, gmm_mu):
        super().__init__()
        self.var_param = nn.Parameter(gmm_mu,
                                      requires_grad=True)

    def forward(self, matrix_a):
        matrix_g = torch.nn.Sigmoid()(self.var_param @ matrix_a)
        return matrix_g

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'var_param = {self.var_param.item()}'


class Uchida(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.var_param = nn.Parameter(w, requires_grad=True)

    def forward(self, matrix_a):
        matrix_g = torch.nn.Sigmoid()(self.var_param @ matrix_a)
        return matrix_g

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'var_param = {self.var_param.item()}'


class RigaDet(nn.Module):
    def __init__(self, weights_size):
        super().__init__()
        self.fc1 = nn.Linear(weights_size, 100, bias=False)
        self.fc2 = nn.Linear(100, 1, bias=False)
        # self.fc3 = nn.Linear(config["watermark_size"], config["watermark_size"], bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.th = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sig(out)
        out = self.fc2(out)
        out = self.sig(out)
        # out = self.fc3(out)
        # out = self.sig(out)
        return out


class RigaExt(nn.Module):
    def __init__(self, config, weights_size):
        super().__init__()
        self.fc1 = nn.Linear(weights_size, 100, bias=False)

        self.fc2 = nn.Linear(100, config["watermark_size"], bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.th = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sig(out)
        out = self.fc2(out)
        out = self.sig(out)
        return out


class EncResistant(nn.Module):
    def __init__(self, config, weight_size):
        super().__init__()

        self.fc1 = nn.Linear(weight_size, 100, bias=True)
        self.fc2 = nn.Linear(100, config["expansion_factor"] * weight_size, bias=True)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.matrix_a = 1. * torch.randn(weight_size * config["expansion_factor"], config["watermark_size"],
                                         requires_grad=False).cuda()


    def forward(self, theta_f):
        out = self.fc1(theta_f)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.sig(out @ self.matrix_a)
        return out



# Custom PolyLU Activation Function it keeps the positives and apply a polynomial transformation to the negatives x/(1 - x) =1 / (1 - x[neg]) - 1
class PolyLU(nn.Module):
    def forward(self, x):
        neg = x < 0
        out = torch.zeros_like(x)
        out[neg] = 1 / (1 - x[neg]) - 1  # negative branch
        out[~neg] = x[~neg]              # non-negative branch
        return out
# Diction normal
class ProjMod(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_channels = config["total_channels"]  # Calculé dynamiquement dans embed
        self.watermark_size = config["watermark_size"]

        # 1. Normalisation Spatiale (Instance Norm est mieux pour les images que LayerNorm ici)
        # Elle normalise chaque map de feature indépendamment
        self.norm = nn.InstanceNorm2d(self.in_channels, affine=True)

        # 2. Bloc Convolutionnel pour traiter la concaténation
        self.conv_block = nn.Sequential(
            # On mélange les canaux concaténés (fusion d'info)
            nn.Conv2d(self.in_channels, 256, kernel_size=3, padding=1, bias=False),

            # nn.ReLU(),
            # nn.Conv2d(64, 512, kernel_size=3, padding=1, bias=False),
            # nn.ReLU(),
            # PolyLU(),
            # AdaptiveAvgPool transforme n'importe quelle taille (H, W) en (1, 1)
            # C'est crucial car "la plus petite taille" peut changer selon le modèle
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 3. Matrice Secrète (Orthogonale)
        matrix_a = torch.randn(self.watermark_size, self.watermark_size)
        # matrix_a, _ = torch.linalg.qr(matrix_a)
        self.register_buffer("matrix_a", matrix_a)

        # 4. Tête de projection finale (Linear)
        # L'entrée est 256 car c'est la sortie du conv_block
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256, bias=False),
            nn.Sigmoid(),
            nn.Linear(256, self.watermark_size, bias=False)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x : [Batch, Total_Channels, H, W]

        # out = self.norm(x)
        out = self.conv_block(x)  # -> [Batch, 256, 1, 1]
        out = self.head(out)  # -> [Batch, watermark_size]

        # Application de la clé secrète et Sigmoid
        # out = out @ self.matrix_a
        out = self.sig(out)

        return out

