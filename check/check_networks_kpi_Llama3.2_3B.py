import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
MODEL_NAME = 'llama-3.2-3b' 
DATASET_NAME = "networks_kpi"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    min_len = min(len(y_true), len(y_pred))
    return np.mean(np.abs(y_true[:min_len] - y_pred[:min_len]))

def check_and_visualize():
    print(f"🔍 [Check Network KPI] Model: {MODEL_NAME}")
    
    file_name = f"results_{DATASET_NAME}_{MODEL_NAME}.pkl"
    
    # Tìm file kết quả (Ưu tiên folder mới)
    path_new = os.path.join(BASE_DIR, f"output/{DATASET_NAME}_{MODEL_NAME}", file_name)
    path_old = os.path.join(BASE_DIR, f"output/{DATASET_NAME}", file_name)
    
    result_path = path_new if os.path.exists(path_new) else (path_old if os.path.exists(path_old) else None)
    
    if not result_path:
        print(f"❌ Không tìm thấy file kết quả!")
        return

    print(f"✅ Đã load file: {result_path}")
    
    img_output_dir = os.path.join(BASE_DIR, f"output/{DATASET_NAME}/{MODEL_NAME}")
    os.makedirs(img_output_dir, exist_ok=True)

    with open(result_path, 'rb') as f:
        results = pickle.load(f)

    data = results['value']
    train = data['train']
    test = data['test']
    pred = data['pred_median']
    samples = data['pred_samples']
    
    mae = calculate_metrics(test.values, pred)
    print(f"   📉 MAE Score: {mae:.4f}")
    
    # --- VẼ BIỂU ĐỒ ---
    plt.figure(figsize=(15, 7))
    
    # Vẽ 3 ngày quá khứ (48 * 3 = 144 điểm) để thấy chu kỳ ngày
    lookback = 144
    plt.plot(range(len(train)-lookback, len(train)), train.iloc[-lookback:], label='History (3 Days)', color='gray', alpha=0.5)
    
    # Vẽ Test & Pred
    x_test = range(len(train), len(train) + len(test))
    plt.plot(x_test, test, label='Ground Truth (KPI)', color='black', linewidth=2)
    plt.plot(x_test[:len(pred)], pred, label='Prediction', color='green', linestyle='--', linewidth=2)
    
    if samples is not None:
        lower = np.percentile(samples, 10, axis=0)
        upper = np.percentile(samples, 90, axis=0)
        plt.fill_between(x_test[:len(lower)], lower, upper, color='green', alpha=0.15, label='Confidence Interval')

    plt.title(f"Network Traffic Forecast ({MODEL_NAME}) - MAE: {mae:.2f}")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    img_path = os.path.join(img_output_dir, "network_kpi_forecast.png")
    plt.savefig(img_path)
    plt.close()
    print(f"   🖼️  Đã lưu biểu đồ: {img_path}")

if __name__ == "__main__":
    check_and_visualize()
