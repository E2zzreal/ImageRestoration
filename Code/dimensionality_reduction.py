import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import numpy as np

# 读取latent features
df = pd.read_csv('latent_features.csv')

# 提取特征和样本信息
X = df.iloc[:, 2:].values  # latent features
samples = df['sample'].values

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# UMAP降维
umap_reducer = umap.UMAP(random_state=42)
X_umap = umap_reducer.fit_transform(X_scaled)

# 计算聚类指标
def calculate_clustering_metrics(X_2d, labels):
    # 检查是否有足够类别
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan  # 返回NaN表示无法计算
    
    # 计算轮廓系数
    try:
        sil_score = silhouette_score(X_2d, labels)
        return sil_score
    except ValueError:
        return np.nan

# 保存结果
results = []
for sample in np.unique(samples):
    # 获取当前sample的索引
    sample_mask = samples == sample
    
    # 计算PCA聚类指标
    pca_score = calculate_clustering_metrics(X_pca[sample_mask], samples[sample_mask])
    
    # 计算UMAP聚类指标
    umap_score = calculate_clustering_metrics(X_umap[sample_mask], samples[sample_mask])
    
    results.append({
        'sample': sample,
        'pca_silhouette_score': pca_score,
        'umap_silhouette_score': umap_score
    })

# 保存降维结果
df_pca = pd.DataFrame(X_pca, columns=['pca_1', 'pca_2'])
df_umap = pd.DataFrame(X_umap, columns=['umap_1', 'umap_2'])
df_results = pd.concat([df[['sample', 'image']], df_pca, df_umap], axis=1)
df_results.to_csv('latent_2d_projections.csv', index=False)

# 保存聚类指标
df_metrics = pd.DataFrame(results)
df_metrics.to_csv('clustering_metrics.csv', index=False)
