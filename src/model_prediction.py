import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import Ridge
import xgboost as xgb
import umap.umap_ as umap
from util import get_standard_scaler, save_results_to_csv

# 读取数据
df_original = pd.read_csv('20240826_ZHmag to TMEC_data-base-5.csv')
df_latent = pd.read_csv('latent_features.csv')

# 提取特征和目标
elements = ['B', 'Al', 'Co', 'Ti', 'Cu', 'Ga', 'Zr', 'Pr', 'Nd']
targets = ['Br: 20', 'Hr: 147']

# UMAP降维到10维
X_latent_raw = df_latent.iloc[:, 2:].values  # 跳过sample和image列
umap_reducer = umap.UMAP(n_components=10, random_state=42)
X_latent_reduced = umap_reducer.fit_transform(X_latent_raw)

# 将降维结果添加回DataFrame
latent_cols = [f'latent_{i}' for i in range(10)]
df_latent_reduced = pd.DataFrame(X_latent_reduced, columns=latent_cols)
df_latent_reduced['sample'] = df_latent['sample']

# 计算每个样本的平均潜在特征
df_latent_mean = df_latent_reduced.groupby('sample')[latent_cols].mean().reset_index()

# 只保留有图片特征的样本
valid_samples = df_latent_mean['sample'].values
df_original_filtered = df_original[df_original['TpNo.'].isin(valid_samples)].copy()

# 确保样本顺序一致
df_original_filtered = df_original_filtered.sort_values('TpNo.')
df_latent_mean = df_latent_mean.sort_values('sample')

# 准备特征数据
X_elements = df_original_filtered[elements].values
X_latent = df_latent_mean[latent_cols].values

# 合并特征
X = np.hstack([X_elements, X_latent])
y = df_original_filtered[targets].values

# 数据标准化
scaler_X = get_standard_scaler()
scaler_y = get_standard_scaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 定义模型和参数网格
models = {
    'XGBoost': (xgb.XGBRegressor(),
                {'n_estimators': [100, 200],
                 'max_depth': [3, 5],
                 'learning_rate': [0.01, 0.1]}),
    
    'SVR': (SVR(),
            {'kernel': ['rbf'],
             'C': [0.1, 1, 10],
             'gamma': ['scale', 'auto']}),
    
    'RandomForest': (RandomForestRegressor(),
                    {'n_estimators': [100, 200],
                     'max_depth': [None, 5, 10]}),
    
    'GaussianProcess': (GaussianProcessRegressor(kernel=C(1.0) * RBF([1.0])),
                       {'alpha': [1e-10, 1e-5]}),
    
    'Ridge': (Ridge(),
             {'alpha': [0.1, 1.0, 10.0]})
}

# 训练和评估模型
results = {}
for target_idx, target_name in enumerate(targets):
    print(f"\n预测 {target_name}:")
    y_train_target = y_train[:, target_idx].reshape(-1, 1)
    y_test_target = y_test[:, target_idx].reshape(-1, 1)
    
    target_results = {}
    for model_name, (model, param_grid) in models.items():
        # 网格搜索优化超参数
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train_target.ravel())
        
        # 预测
        y_pred = grid_search.predict(X_test).reshape(-1, 1)
        
        # 正确的反标准化
        y_test_padded = np.zeros((len(y_test_target), 2))
        y_pred_padded = np.zeros((len(y_pred), 2))
        
        y_test_padded[:, target_idx] = y_test_target.ravel()
        y_pred_padded[:, target_idx] = y_pred.ravel()
        
        y_test_original = scaler_y.inverse_transform(y_test_padded)[:, target_idx]
        y_pred_original = scaler_y.inverse_transform(y_pred_padded)[:, target_idx]
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        r2 = r2_score(y_test_original, y_pred_original)
        
        target_results[model_name] = {'RMSE': rmse, 'R2': r2}
        print(f"\n{model_name}:")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
    
    results[target_name] = target_results

# 保存结果
results_df = pd.DataFrame.from_dict({(i,j): results[i][j] 
                                   for i in results.keys() 
                                   for j in results[i].keys()}, 
                                   orient='index')
save_results_to_csv(results_df, 'prediction_results.csv')
