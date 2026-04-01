import pandas as pd    
import numpy as np    
import time  
from sklearn.cluster import KMeans    
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
  
df = pd.read_csv("input/machine-train1-1.csv", sep=",")  
df = df.drop(df.columns[0], axis=1)    
df = df.astype(float)   
print(df.shape)   

scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)

window_size = 32
step_size = 15

train_windows = []
for i in range(0, len(df_normalized) - window_size + 1, step_size):
    window_data = df_normalized[i:i+window_size]
    window_means = window_data.mean(axis=0).reshape(-1, 1)
    train_windows.append(window_means)

num_clusters = 10
all_cluster_centers = []
all_cluster_labels = []

for window_idx, window_means in enumerate(train_windows):
    if len(window_means) >= num_clusters:
        kmeans = KMeans(n_clusters=num_clusters, max_iter=1000, random_state=42)    
        cluster_labels = kmeans.fit_predict(window_means)    
        cluster_centers = kmeans.cluster_centers_
        
        cluster_thresholds = []
        for k in range(num_clusters):
            cluster_indices = np.where(cluster_labels == k)[0]
            if len(cluster_indices) > 0:
                cluster_values = window_means[cluster_indices]
                distances = np.abs(cluster_values - cluster_centers[k])
                alpha_k = np.mean(distances)
                cluster_thresholds.append(alpha_k)
            else:
                cluster_thresholds.append(0.0)
        
        all_cluster_centers.append((cluster_centers, cluster_thresholds))
        all_cluster_labels.append(cluster_labels)
    else:
        all_cluster_centers.append((np.zeros((num_clusters, 1)), [0.0]*num_clusters))
        all_cluster_labels.append(np.zeros(len(window_means)))

train_speed_thresholds = []
for i in range(0, len(df_normalized) - window_size + 1, step_size):
    window_data = df_normalized[i:i+window_size]
    window_speeds = np.abs(np.diff(window_data, axis=0))
    
    var_thresholds = []
    for var_idx in range(window_data.shape[1]):
        speeds = window_speeds[:, var_idx]
        if len(speeds) > 1:
     
            confidence = 0.95
            n = len(speeds)
            mean_speed = np.mean(speeds)
            std_speed = np.std(speeds, ddof=1)
            
            t_value = stats.t.ppf((1 + confidence) / 2, n-1)
            margin_error = t_value * std_speed / np.sqrt(n)
            
            beta_l = max(0, mean_speed - margin_error)
            beta_u = mean_speed + margin_error
            
            var_thresholds.append((beta_l, beta_u))
        else:
            var_thresholds.append((0.0, 0.1))  
    
    train_speed_thresholds.append(var_thresholds)
  
anomaly_data = np.load("save-machine-testlabel-1-1.npy", allow_pickle=True)   
anomaly_data = anomaly_data.astype(float)  
print(anomaly_data.shape)

anomaly_data_normalized = scaler.transform(anomaly_data)
  
anomaly_flags = np.zeros(anomaly_data.shape, dtype=int)

f_values = np.zeros(anomaly_data.shape, dtype=float)

lambda1 = 0.5  
lambda2 = 0.5  

st = time.time()

num_windows = min(len(all_cluster_centers), 
                  (len(anomaly_data_normalized) - window_size) // step_size + 1)

for window_idx in range(num_windows):
    start_idx = window_idx * step_size
    end_idx = min(start_idx + window_size, len(anomaly_data_normalized))
    
    if end_idx <= start_idx:
        continue
    
    window_data = anomaly_data_normalized[start_idx:end_idx]
    
    cluster_centers, cluster_thresholds = all_cluster_centers[window_idx]
    speed_thresholds = train_speed_thresholds[window_idx]
    
    if window_idx < len(all_cluster_labels):
        cluster_labels_window = all_cluster_labels[window_idx]
        column_to_cluster_label = {col: label for col, label in enumerate(cluster_labels_window)}
    else:
        column_to_cluster_label = {col: col % num_clusters for col in range(window_data.shape[1])}
    
    for i in range(len(window_data)):
        global_idx = start_idx + i
        
        for var_idx in range(window_data.shape[1]):
            current_value = window_data[i, var_idx]
            
            cluster_label = column_to_cluster_label.get(var_idx, 0)
            cluster_center = cluster_centers[cluster_label, 0] if cluster_label < len(cluster_centers) else 0.5
            alpha_k = cluster_thresholds[cluster_label] if cluster_label < len(cluster_thresholds) else 0.1
            
            var_distance = np.abs(current_value - cluster_center)
            var_violation = var_distance - alpha_k
            
            time_violation = 0.0
            if i > 0: 
                prev_value = window_data[i-1, var_idx]
                speed = np.abs(current_value - prev_value)
                
                beta_l, beta_u = speed_thresholds[var_idx] if var_idx < len(speed_thresholds) else (0.0, 0.1)
                
                if speed < beta_l:
                    time_violation = beta_l - speed 
                elif speed > beta_u:
                    time_violation = speed - beta_u 
                else:
                    time_violation = 0.0 
            
            if var_violation > 0 or time_violation > 0:
                chi = lambda1 * max(var_violation, 0) + lambda2 * time_violation
            else:
                chi = lambda1 * var_violation + lambda2 * time_violation
            
            if global_idx == 0:
                f_values[global_idx, var_idx] = chi
            else:
                f_values[global_idx, var_idx] = f_values[global_idx-1, var_idx] + chi
            
            if global_idx == 0:
                if chi > 0:
                    anomaly_flags[global_idx, var_idx] = 1
            else:
                if f_values[global_idx, var_idx] > f_values[global_idx-1, var_idx]:
                    anomaly_flags[global_idx, var_idx] = 1
                else:
                    anomaly_flags[global_idx, var_idx] = 0

ed = time.time()

#print(f"time: {ed - st:.2f}s")

d_dirty = np.array(anomaly_data)  
d_clean = np.array(df)  
bound = (np.max(d_dirty, axis=0) - np.max(d_clean, axis=0)) / 10000
true_errors = (d_dirty - d_clean >= bound).astype(int)  

tp = np.logical_and(anomaly_flags, true_errors)  
fp = np.logical_and(anomaly_flags, 1-true_errors)    
fn = np.logical_and(1-anomaly_flags, true_errors)  
tn = np.logical_and(1-anomaly_flags, 1-true_errors)  

tps = np.sum(tp)  
fps = np.sum(fp)  
fns = np.sum(fn)  
tns = np.sum(tn)  

print('tp:', tps)  
print('fp:', fps)  
print('fn:', fns)  
print('tn:', tns)  

P = np.sum(tp) / (np.sum(tp) + np.sum(fp)) if (np.sum(tp) + np.sum(fp)) > 0 else 0    
R = np.sum(tp) / (np.sum(tp) + np.sum(fn)) if (np.sum(tp) + np.sum(fn)) > 0 else 0    
F1 = (2 * P * R) / (P + R) if (P + R) > 0 else 0  

print('精确率(P):', P)  
print('召回率(R):', R)  
print('F1分数:', F1)  
print('总时间成本:', ed - st)
