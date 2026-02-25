import torch.nn as nn


class HufuNet(nn.Module):
    """
    Optimal MNIST Autoencoder based on best practices and empirical results

    Architecture Details:
    - Input: 28x28x1 MNIST images
    - Latent dimension: 64 (good balance between compression and quality)
    - Uses BatchNorm for stable training
    - Uses LeakyReLU for better gradient flow
    - Progressive downsampling/upsampling
    """

    def __init__(self, latent_dim=32):
        super(HufuNet, self).__init__()

        # ENCODER
        self.encoder = nn.Sequential(
            # First conv block: 28x28x1 -> 14x14x16
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28->14
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # Second conv block: 14x14x16 -> 7x7x32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 14->7
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Third conv block: 7x7x32 -> 4x4x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 7->4 (with padding)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Flatten and reduce to latent dimension
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),  # 1024 -> 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim),  # 128 -> 32 (latent space)
        )
        # DECODER
        self.decoder = nn.Sequential(
            # Expand from latent dimension
            nn.Linear(latent_dim, 128),  # 32 -> 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64 * 4 * 4),  # 512 -> 2048
            nn.LeakyReLU(0.2, inplace=True),

            # Reshape to feature maps
            nn.Unflatten(1, (64, 4, 4)),  # Reshape to 128x4x4

            # First deconv block: 4x4x128 -> 7x7x64
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),  # 4->7
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Second deconv block: 7x7x64 -> 14x14x32
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7->14
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # Final deconv block: 14x14x32 -> 28x28x1
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14->28
            nn.Sigmoid()  # Output in [0,1] range
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)

    def decode(self, z):
        """Reconstruct from latent representation"""
        return self.decoder(z)
