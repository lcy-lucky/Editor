import pandas as pd  
import numpy as np  
from sklearn.cluster import KMeans  
  
# 读取数据并预处理  
df = pd.read_csv("input/machine-train1-1-1.csv", sep=",")
df = df.drop(df.columns[0], axis=1)  
df = df.astype(float) 
print(df.shape) 
  
# 计算每列的平均值  
column_means = df.mean(axis=0).values.reshape(-1, 1)  
  
# 对列的平均值进行聚类  
kmeans = KMeans(n_clusters=10, max_iter=1000)  
cluster_labels = kmeans.fit_predict(column_means)  
  
# 聚类中心  
cluster_centers = kmeans.cluster_centers_  
  
# 加载异常数据集  
anomaly_data = np.load("2_modified_smd_dataset_copy.npy", allow_pickle=True)  
anomaly_data = anomaly_data.astype(float)  
print(anomaly_data.shape)
  
# 初始化异常标记矩阵，与anomaly_data形状相同，初始值为0  
anomaly_flags = np.zeros(anomaly_data.shape, dtype=int)  
  
# 设定异常检测的阈值倍数  
threshold_multiplier = 3  
  
original_columns = df.columns  # 保存原始列名  
# dropped_columns = df.columns[[0, 45]]  # 被删除的列名  
remaining_columns = original_columns
  
# 创建一个从remaining_columns到聚类标签的映射  
column_to_cluster_label = {col: label for col, label in zip(remaining_columns, cluster_labels)}   

# ...（异常检测逻辑，但使用column_to_cluster_label来查找聚类标签）  
for i in range(anomaly_data.shape[0]):  
    for j, col in enumerate(remaining_columns):  # 使用remaining_columns的索引和名称  
        cluster_idx = column_to_cluster_label[col]  
        cluster_center_value = cluster_centers[cluster_idx, 0]
          
        # 获取当前数据点及其前后数据点（如果存在）  
        current_value = anomaly_data[i, j]  
        prev_value = anomaly_data[i-1, j] if i > 0 else np.nan  
        next_value = anomaly_data[i+1, j] if i < anomaly_data.shape[0] - 1 else np.nan  
          
        # 计算与聚类中心的差异以及与前后数据点的差异  
        center_diff = np.abs(current_value - cluster_center_value)  
        prev_diff = np.abs(current_value - prev_value) if not np.isnan(prev_value) else 0  
        next_diff = np.abs(current_value - next_value) if not np.isnan(next_value) else 0  
          
        # 判断是否异常  
        # if (center_diff >= threshold_multiplier * cluster_center_value and  
        #     (prev_diff >= threshold_multiplier * cluster_center_value or np.isnan(prev_value)) and  
        #     (next_diff >= threshold_multiplier * cluster_center_value or np.isnan(next_value))):
        if ((center_diff >= threshold_multiplier * cluster_center_value) and  
            (prev_diff >= threshold_multiplier * cluster_center_value) and  
            (next_diff >= threshold_multiplier * cluster_center_value)):  
            anomaly_flags[i, j] = 1  
  
d_dirty = np.array(anomaly_data)
d_clean = np.array(df)
d_repair = np.array(anomaly_flags)
# bound = (np.max(d_dirty, axis=0) - np.max(d_clean, axis=0)) / 1000
# bound = (np.max(d_dirty, axis=0) - np.min(d_dirty, axis=0)) / 10000
bound = (np.max(d_dirty, axis=0) - np.min(d_dirty, axis=0)) / 10
true_errors = (d_dirty - d_clean > bound).astype(int)
# true_errors = (d_dirty - d_clean > bound).astype(int)

 # 计算TP（真正例）：既被检测到也是真实错误的点  
tp = np.logical_and(d_repair, true_errors)
      
    # 计算FP（假正例）：被检测到但不是真实错误的点  
fp = np.logical_and(d_repair, 1-true_errors)  
      
    # 计算FN（假负例）：未被检测到但实际上是错误的点  
fn = np.logical_and(1-d_repair, true_errors)
    # tp = d_repair * true_errors  
tn = np.logical_and(1-d_repair, 1-true_errors)
first_row_d_repair = d_repair[0, :] 
print('d_repair', first_row_d_repair)
row_1_d_repair = 1-d_repair
first_row_1_d_repair = row_1_d_repair[0, :]
print('1-d_repair', first_row_1_d_repair)
tps=np.sum(tp)
fps=np.sum(fp)
fns=np.sum(fn)
tns=np.sum(tn)

print('tp:', tps)
print('fp:', fps)
print('fn:', fns)
print('tn:', tns)
    # 计算准确率和召回率  
P = np.sum(tp) / (np.sum(tp) + np.sum(fp)) if (np.sum(tp) + np.sum(fp)) > 0 else 0  
R = np.sum(tp) / (np.sum(tp) + np.sum(fn)) if (np.sum(tp) + np.sum(fn)) > 0 else 0  
F1 = (P * R) / (P + R) * 2 
print('P', P)
print('R', R)
print('F1', F1)

