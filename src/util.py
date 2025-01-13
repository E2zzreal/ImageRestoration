import torch
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

def get_device():
    """获取可用设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_image_transform():
    """获取图像预处理transform"""
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

def get_standard_scaler():
    """获取标准化scaler"""
    return StandardScaler()

def ensure_dir_exists(dir_path):
    """确保目录存在"""
    os.makedirs(dir_path, exist_ok=True)

def save_results_to_csv(data, file_path, index=False):
    """保存结果到CSV文件"""
    ensure_dir_exists(os.path.dirname(file_path))
    if isinstance(data, pd.DataFrame):
        data.to_csv(file_path, index=index)
    else:
        pd.DataFrame(data).to_csv(file_path, index=index)

def save_model(model, path):
    """保存模型"""
    ensure_dir_exists(os.path.dirname(path))
    torch.save(model.state_dict(), path)
