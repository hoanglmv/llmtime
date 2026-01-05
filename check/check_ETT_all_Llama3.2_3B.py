import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
MODEL_NAME = 'llama-3.2-3b' 
DATASETS_TO_CHECK = ["ETTm1", "ETTm2", "ETTh1", "ETTh2"]

# Đường dẫn gốc project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    return mae, mse

def check_and_visualize():
    print(f"🔍 [Check] Model: {MODEL_NAME}")
    
    for ds_name in DATASETS_TO_CHECK:
        print(f"\n" + "="*60)
        print(f"📊 DATASET: {ds_name}")
        
        # --- LOGIC TÌM FILE THÔNG MINH ---
        file_name = f"results_{ds_name}_{MODEL_NAME}.pkl"
        
        # Cách 1: Tìm ở đường dẫn cũ (output/ETTm1/...)
        path_old = os.path.join(BASE_DIR, f"output/{ds_name}", file_name)
        
        # Cách 2: Tìm ở đường dẫn mới (output/ETTm1_llama-3.2-3b/...)
        path_new = os.path.join(BASE_DIR, f"output/{ds_name}_{MODEL_NAME}", file_name)
        
        result_file = None
        if os.path.exists(path_new):
            result_file = path_new
            print(f"   ✅ Tìm thấy file tại (Mới): {path_new}")
        elif os.path.exists(path_old):
            result_file = path_old
            print(f"   ✅ Tìm thấy file tại (Cũ): {path_old}")
        else:
            print(f"   ❌ Không tìm thấy file ở cả 2 nơi:")
            print(f"      - {path_new}")
            print(f"      - {path_old}")
            continue

        # 2. ĐỊNH NGHĨA ĐƯỜNG DẪN LƯU ẢNH
        image_output_dir = os.path.join(BASE_DIR, f"output/{ds_name}/{MODEL_NAME}")
        os.makedirs(image_output_dir, exist_ok=True)
        print(f"   📂 Thư mục lưu ảnh: {image_output_dir}")

        try:
            with open(result_file, 'rb') as f:
                results = pickle.load(f)
        except Exception as e:
            print(f"   ❌ Lỗi đọc file pickle: {e}")
            continue

        for col, data in results.items():
            try:
                train = data['train']
                test = data['test']
                pred = data['pred_median']
                samples = data['pred_samples']
                
                # Tính Metrics
                mae, mse = calculate_metrics(test.values, pred)
                print(f"   📍 {col}: MAE={mae:.4f} | MSE={mse:.4f}")
                
                # Vẽ biểu đồ
                plt.figure(figsize=(12, 6))
                
                # Vẽ lịch sử (100 điểm cuối)
                lookback = 100 
                if len(train) > lookback:
                    plt.plot(range(len(train)-lookback, len(train)), train.iloc[-lookback:], label='History', color='gray', alpha=0.5)
                else:
                    plt.plot(range(len(train)), train, label='History', color='gray', alpha=0.5)
                
                x_test = range(len(train), len(train) + len(test))
                plt.plot(x_test, test, label='Ground Truth', color='black', linewidth=2)
                
                plot_len = min(len(pred), len(x_test))
                plt.plot(x_test[:plot_len], pred[:plot_len], label='Prediction', color='red', linestyle='--')
                
                # Vẽ vùng tin cậy
                if samples is not None and len(samples) > 0:
                    samples = np.array(samples)
                    lower = np.percentile(samples, 10, axis=0)
                    upper = np.percentile(samples, 90, axis=0)
                    sample_len = min(len(lower), plot_len)
                    plt.fill_between(x_test[:sample_len], lower[:sample_len], upper[:sample_len], color='red', alpha=0.2)

                plt.title(f"{ds_name} - {col} ({MODEL_NAME}) | MAE: {mae:.2f}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Lưu ảnh
                img_path = os.path.join(image_output_dir, f"{col}.png")
                plt.savefig(img_path)
                plt.close()
                
            except Exception as e:
                print(f"      ❌ Lỗi xử lý cột {col}: {e}")

    print("\n🎉 HOÀN TẤT CHECK MODEL 1!")

if __name__ == "__main__":
    check_and_visualize()