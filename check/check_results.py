import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "output")
MODEL_NAME = 'gpt-4' 
DATASETS = ["ETTm1"] # Thêm ETTm2 nếu bạn đã chạy nó

def draw_plots():
    print(f"📊 Đang vẽ biểu đồ từ kết quả model: {MODEL_NAME}")
    
    for ds_name in DATASETS:
        file_path = os.path.join(OUTPUT_BASE_DIR, ds_name, f"results_{ds_name}_{MODEL_NAME}.pkl")
        
        if not os.path.exists(file_path):
            print(f"❌ Chưa có kết quả cho {ds_name}. Hãy chạy model trước!")
            continue
            
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
            
        # Tạo thư mục lưu ảnh
        save_dir = os.path.join(OUTPUT_BASE_DIR, ds_name, f"plots_{MODEL_NAME}_new")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"   📂 Dataset {ds_name}: Tìm thấy {len(results)} cột.")
        
        for col, data in results.items():
            train = data['train']
            test = data['test']
            pred = pd.Series(data['pred_median'], index=test.index)
            samples = data['pred_samples']
            
            # Tính sai số
            mae = np.mean(np.abs(pred - test))
            
            # Vẽ hình
            plt.figure(figsize=(12, 6))
            
            # Vẽ 300 điểm lịch sử cuối
            plt.plot(train.index[-300:], train.values[-300:], label='History', color='gray', alpha=0.5)
            plt.plot(test.index, test.values, label='Ground Truth', color='black', linewidth=2)
            plt.plot(test.index, pred.values, label='GPT-4 Prediction', color='#1f77b4', linestyle='--', linewidth=2)
            
            # Vẽ khoảng tin cậy
            if samples is not None:
                try:
                    if isinstance(samples, list):
                        vals = np.array([s.values for s in samples])
                    else:
                        vals = samples
                    lower = np.quantile(vals, 0.05, axis=0)
                    upper = np.quantile(vals, 0.95, axis=0)
                    plt.fill_between(test.index, lower, upper, color='#1f77b4', alpha=0.15)
                except: pass

            plt.title(f"GPT-4 Forecast: {ds_name} - {col}\nMAE: {mae:.2f}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            img_path = os.path.join(save_dir, f"{col}.png")
            plt.savefig(img_path)
            plt.close()
            print(f"      ✅ Đã vẽ: {col} -> MAE={mae:.2f}")

    print(f"\n✨ Xong! Ảnh được lưu tại: {os.path.join(OUTPUT_BASE_DIR, 'ETTm1', f'plots_{MODEL_NAME}_new')}")

if __name__ == "__main__":
    draw_plots()