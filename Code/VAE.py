import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import DatasetFolder
from torchvision import models
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from PIL import Image

import os
import numpy as np
import pandas as pd


#定义VAE（卷积层）
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 示例中假设输入是单通道图像
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(512, 64)  # 假设最终的特征图大小为 25x25x128
        self.fc_logvar = nn.Linear(512, 64)

        # 解码器
        self.decoder_fc1 = nn.Linear(64, 512)
        self.decoder_fc2 = nn.Linear(512, 128 * 16 * 16)
        self.relu = nn.ReLU()
        self.decoder = nn.Sequential(
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
        h = h.view(-1, 128, 16, 16)  # 重塑张量以匹配解码器的输入维度
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


#定义损失函数
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')

    # KL散度损失
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

#训练循环
def train(model, num_epochs,dataloader, device, optimizer):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in dataloader:
            #print (data.shape)
            data = data[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            #print(recon_batch.shape)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #print(loss)
        # 记录每个epoch的平均损失
        epoch_loss /= len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss}', flush=True)
        
        if (epoch + 1) % 500 == 0:
            # 指定保存模型的路径和文件名
            save_path = f'./modelsL3e-4/model_epoch_{epoch+1}.pth'
            # 保存模型状态字典
            torch.save(model.state_dict(), save_path)
            print(f'Model saved at {save_path}')



# 数据准备
transform = transforms.Compose([
#     transforms.CenterCrop(size=(500,500)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
# #    transforms.RandomRotation(degrees=180, expand=True, center=None),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    ])


if __name__ == '__main__':

    # 确保保存模型的目录存在
    os.makedirs('./modelsL3e-4', exist_ok=True)
    # 训练模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ConvVAE().to(device)
    learning_rate = 3e-4
    batch_size = 2048
    epochs = 10000

    dataset = datasets.ImageFolder(root=r'2-Data/2-SEM/X5000/AugmentationL', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train(model, epochs, dataloader, device, optimizer)
