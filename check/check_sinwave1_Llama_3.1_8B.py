import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
MODEL_NAME = 'llama-3.1-8b' 
DATASET_NAME = "sin_wave_var"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    min_len = min(len(y_true), len(y_pred))
    return np.mean(np.abs(y_true[:min_len] - y_pred[:min_len]))

def check_and_visualize():
    print(f"🔍 [Check SinWave Variable] Model: {MODEL_NAME}")
    
    # Logic tìm file thông minh
    file_name = f"results_{DATASET_NAME}_{MODEL_NAME}.pkl"
    
    # 1. Tìm ở folder mới (có suffix tên model)
    path_new = os.path.join(BASE_DIR, f"output/{DATASET_NAME}_{MODEL_NAME}", file_name)
    # 2. Tìm ở folder cũ (không suffix)
    path_old = os.path.join(BASE_DIR, f"output/{DATASET_NAME}", file_name)
    
    result_path = None
    if os.path.exists(path_new):
        result_path = path_new
    elif os.path.exists(path_old):
        result_path = path_old
    
    if not result_path:
        print(f"❌ Không tìm thấy file kết quả!")
        print(f"   - Đã tìm: {path_new}")
        print(f"   - Đã tìm: {path_old}")
        return

    print(f"✅ Đã load file: {result_path}")
    
    # Folder lưu ảnh riêng cho model 8B
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
    
    # Vẽ 1000 điểm Train cuối cùng để thấy rõ chu kỳ biến thiên dài
    lookback = 1000
    plt.plot(range(len(train)-lookback, len(train)), train.iloc[-lookback:], label='History (Llama 8B Context)', color='gray', alpha=0.5)
    
    # Vẽ Test & Pred
    x_test = range(len(train), len(train) + len(test))
    plt.plot(x_test, test, label='Ground Truth', color='black', linewidth=2)
    plt.plot(x_test[:len(pred)], pred, label='Prediction (8B)', color='blue', linestyle='--', linewidth=2) # Đổi màu xanh cho khác biệt
    
    # Vẽ vùng tin cậy
    if samples is not None:
        lower = np.percentile(samples, 10, axis=0)
        upper = np.percentile(samples, 90, axis=0)
        plt.fill_between(x_test[:len(lower)], lower, upper, color='blue', alpha=0.15, label='Confidence Interval')

    plt.title(f"Variable Amplitude Sin Wave (Llama 3.1 8B) - MAE: {mae:.2f}")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    img_path = os.path.join(img_output_dir, "variable_sin_wave_8B.png")
    plt.savefig(img_path)
    plt.close()
    print(f"   🖼️  Đã lưu biểu đồ tại: {img_path}")

if __name__ == "__main__":
    check_and_visualize()
