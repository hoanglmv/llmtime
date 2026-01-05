import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
MODEL_NAME = 'llama-3.2-3b' 
DATASET_NAME = "sin_wave_var" # Tên dataset mới

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
    result_path = os.path.join(BASE_DIR, f"output/{DATASET_NAME}_{MODEL_NAME}", file_name)
    
    if not os.path.exists(result_path):
        print(f"❌ Không tìm thấy file: {result_path}")
        return

    print(f"✅ Đã load file: {result_path}")
    
    # Folder lưu ảnh
    img_output_dir = os.path.join(BASE_DIR, f"output/{DATASET_NAME}/{MODEL_NAME}")
    os.makedirs(img_output_dir, exist_ok=True)

    with open(result_path, 'rb') as f:
        results = pickle.load(f)

    # Dataset này chỉ có cột 'value'
    data = results['value']
    train = data['train']
    test = data['test']
    pred = data['pred_median']
    samples = data['pred_samples']
    
    mae = calculate_metrics(test.values, pred)
    print(f"   📉 MAE Score: {mae:.4f}")
    
    # --- VẼ BIỂU ĐỒ ---
    plt.figure(figsize=(15, 7)) # Vẽ to hơn chút để nhìn rõ sóng
    
    # Vẽ 500 điểm Train cuối cùng (để thấy được sự biến thiên biên độ trước đó)
    lookback = 1000
    plt.plot(range(len(train)-lookback, len(train)), train.iloc[-lookback:], label='History (Variable Amp)', color='gray', alpha=0.5)
    
    # Vẽ Test & Pred
    x_test = range(len(train), len(train) + len(test))
    plt.plot(x_test, test, label='Ground Truth', color='black', linewidth=2)
    plt.plot(x_test[:len(pred)], pred, label='Prediction', color='red', linestyle='--', linewidth=2)
    
    # Vẽ vùng tin cậy
    if samples is not None:
        lower = np.percentile(samples, 10, axis=0)
        upper = np.percentile(samples, 90, axis=0)
        plt.fill_between(x_test[:len(lower)], lower, upper, color='red', alpha=0.15, label='Confidence Interval')

    plt.title(f"Variable Amplitude Sin Wave - MAE: {mae:.2f}\n(Chu kỳ hẹp + Biên độ thay đổi)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    img_path = os.path.join(img_output_dir, "variable_sin_wave.png")
    plt.savefig(img_path)
    plt.close()
    print(f"   🖼️  Đã lưu biểu đồ tại: {img_path}")
    print("   👉 Mở ảnh lên để xem model có bắt được nhịp 'sóng to sóng nhỏ' không nhé!")

if __name__ == "__main__":
    check_and_visualize()
