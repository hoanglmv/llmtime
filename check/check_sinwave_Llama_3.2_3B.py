import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
MODEL_NAME = 'llama-3.2-3b' 
DATASET_NAME = "sin_wave"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    min_len = min(len(y_true), len(y_pred))
    return np.mean(np.abs(y_true[:min_len] - y_pred[:min_len]))

def check_and_visualize():
    print(f"🔍 [Check SinWave] Model: {MODEL_NAME}")
    
    # Tìm file kết quả
    file_name = f"results_{DATASET_NAME}_{MODEL_NAME}.pkl"
    # Ưu tiên tìm ở folder có suffix tên model
    result_path = os.path.join(BASE_DIR, f"output/{DATASET_NAME}_{MODEL_NAME}", file_name)
    
    if not os.path.exists(result_path):
        # Fallback tìm ở folder thường
        result_path = os.path.join(BASE_DIR, f"output/{DATASET_NAME}", file_name)
    
    if not os.path.exists(result_path):
        print(f"❌ Không tìm thấy file kết quả tại: {result_path}")
        return

    print(f"✅ Đã load file: {result_path}")
    
    # Folder lưu ảnh
    img_output_dir = os.path.join(BASE_DIR, f"output/{DATASET_NAME}/{MODEL_NAME}")
    os.makedirs(img_output_dir, exist_ok=True)

    with open(result_path, 'rb') as f:
        results = pickle.load(f)

    for col, data in results.items():
        train = data['train']
        test = data['test']
        pred = data['pred_median']
        samples = data['pred_samples']
        
        mae = calculate_metrics(test.values, pred)
        print(f"   📍 Cột '{col}': MAE = {mae:.4f}")
        
        # VẼ BIỂU ĐỒ
        plt.figure(figsize=(12, 6))
        
        # Vẽ 200 điểm Train cuối cùng
        lookback = 200
        plt.plot(range(len(train)-lookback, len(train)), train.iloc[-lookback:], label='History', color='gray', alpha=0.5)
        
        # Vẽ Test & Pred
        x_test = range(len(train), len(train) + len(test))
        plt.plot(x_test, test, label='Ground Truth (Sin Wave)', color='black', linewidth=2)
        plt.plot(x_test[:len(pred)], pred, label='Llama Prediction', color='red', linestyle='--')
        
        # Vẽ vùng tin cậy
        if samples is not None:
            lower = np.percentile(samples, 10, axis=0)
            upper = np.percentile(samples, 90, axis=0)
            plt.fill_between(x_test[:len(lower)], lower, upper, color='red', alpha=0.2)

        plt.title(f"Sin Wave Forecast (Amp~1000) - MAE: {mae:.2f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        img_path = os.path.join(img_output_dir, f"{col}.png")
        plt.savefig(img_path)
        plt.close()
        print(f"   🖼️  Đã lưu ảnh: {img_path}")

if __name__ == "__main__":
    check_and_visualize()
