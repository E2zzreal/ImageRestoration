import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from util import get_device, get_image_transform, save_model

# 定义VAE（卷积层）
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 2048),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(2048, 64)
        self.fc_logvar = nn.Linear(2048, 64)

        # 解码器
        self.decoder_fc1 = nn.Linear(64, 2048)
        self.decoder_fc2 = nn.Linear(2048, 256 * 16 * 16)
        self.relu = nn.ReLU()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.relu(self.decoder_fc1(z))
        h = self.relu(self.decoder_fc2(h))
        h = h.view(-1, 256, 16, 16)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 定义损失函数
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 训练循环
def train(model, num_epochs, dataloader, device, optimizer):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in dataloader:
            data = data[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss}', flush=True)
        
        if (epoch + 1) % 100 == 0:
            save_path = f'./modelsVL5e-4/model_epoch_{epoch+1}.pth'
            save_model(model, save_path)
            print(f'Model saved at {save_path}')

if __name__ == '__main__':
    # 初始化
    device = get_device()
    transform = get_image_transform()
    
    # 模型参数
    model = ConvVAE().to(device)
    learning_rate = 5e-4
    batch_size = 392
    epochs = 5000

    # 数据加载
    dataset = datasets.ImageFolder(root=r'2-Data/2-SEM/X5000/AugmentationVL', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train(model, epochs, dataloader, device, optimizer)
