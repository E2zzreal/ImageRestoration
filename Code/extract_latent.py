import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import os
from PIL import Image
import numpy as np

# 定义与训练时相同的VAE模型结构
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        # Encoder
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
        
        # Decoder
        self.decoder_fc1 = nn.Linear(64, 2048)
        self.decoder_fc2 = nn.Linear(2048, 256 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h)

# 图片预处理
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvVAE().to(device)
model_path = '/home/kemove/Desktop/Mag/2-ZH/0-ImageProcessing/modelsVL5e-4/model_epoch_5000.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 创建数据集
dataset = datasets.ImageFolder(root='/home/kemove/Desktop/Mag/2-ZH/0-ImageProcessing/2-Data/2-SEM/X5000/AugmentationVL', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 提取隐变量并保存
results = []
with torch.no_grad():
    for i, (img, _) in enumerate(dataloader):
        img = img.to(device)
        latent = model.encode(img).cpu().numpy().flatten()
        
        # 解析图片路径
        img_path = dataset.imgs[i][0]
        img_name = os.path.basename(img_path)
        sample_num, img_num = img_name.split('-')[0], img_name.split('-')[1].split('x')[0]
        
        # 创建结果行
        row = {'sample': sample_num, 'image': img_num}
        for j in range(64):
            row[f'latent_{j}'] = latent[j]
        results.append(row)

# 保存为CSV
df = pd.DataFrame(results)
df.to_csv('latent_features.csv', index=False)
