# ImageRestoration

## 项目概述
本项目用于图像特征提取、降维和预测分析。通过VAE模型提取图像潜在特征，使用PCA和UMAP进行降维，并建立回归模型预测材料性能。

## 项目结构
```
ImageRestoration/
├── notebooks/                # Jupyter notebooks
├── Results/                  # 结果文件
│   ├── clustering_metrics.csv
│   ├── latent_2d_projections.csv
│   ├── latent_features.csv
│   └── Images/               # 生成图像
├── src/                      # 源代码
│   ├── dimensionality_reduction.py  # 降维分析
│   ├── extract_latent.py     # 潜在特征提取
│   ├── model_prediction.py   # 模型预测
│   ├── VAEVL.py              # VAE模型
│   └── util.py               # 工具函数
├── requirements.txt          # 依赖库
└── README.md                 # 项目说明
```

## 主要功能
- 图像特征提取：使用VAE模型提取图像潜在特征
- 数据降维：支持PCA和UMAP降维方法
- 性能预测：建立回归模型预测材料性能
- 结果可视化：生成2D投影图和聚类指标

## 使用方法
1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 提取图像特征：
```bash
python src/extract_latent.py
```

3. 进行降维分析：
```bash
python src/dimensionality_reduction.py
```

4. 运行预测模型：
```bash
python src/model_prediction.py
```

## 依赖库
- torch
- torchvision
- pandas
- numpy
- scikit-learn
- umap-learn
- xgboost

## 工具模块说明
`util.py` 提供以下工具函数：
- `get_image_transform()`: 获取图像预处理transform
- `get_standard_scaler()`: 获取标准化scaler
- `save_results_to_csv()`: 保存结果到CSV文件

## 示例
```python
from util import get_image_transform, save_results_to_csv

# 获取图像预处理
transform = get_image_transform()

# 保存结果
results = {'sample': [1, 2], 'value': [0.5, 0.8]}
save_results_to_csv(results, 'output.csv')
