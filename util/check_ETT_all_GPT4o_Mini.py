import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.expanduser("/home/myvh07/hoanglmv/Project/llmtime")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "output")

DATASETS = ["ETTm1", "ETTm2", "ETTh1", "ETTh2"]

# Tên model dùng trong tên file kết quả
MODEL_NAME_TAG = "gpt-4o-mini"

def check_gpt4o_mini_results():
    print(f"📂 Đang kiểm tra kết quả tại: {OUTPUT_BASE_DIR}")
    
    summary_data = []

    for ds_name in DATASETS:
        # Đường dẫn file kết quả
        result_file = os.path.join(OUTPUT_BASE_DIR, ds_name, f"results_{ds_name}_{MODEL_NAME_TAG}.pkl")
        
        if not os.path.exists(result_file):
            print(f"⚠️ Không tìm thấy file: {result_file}. Bỏ qua.")
            continue
            
        print(f"\n" + "="*50)
        print(f"📊 PHÂN TÍCH: {ds_name} ({MODEL_NAME_TAG})")
        print("="*50)
        
        try:
            with open(result_file, 'rb') as f:
                all_results = pickle.load(f)
        except Exception as e:
            print(f"❌ Lỗi đọc file pickle: {e}")
            continue
        
        # Thư mục lưu ảnh
        img_dir = os.path.join(OUTPUT_BASE_DIR, ds_name, f"plots_{MODEL_NAME_TAG}")
        os.makedirs(img_dir, exist_ok=True)
        
        for col, data in all_results.items():
            train = data['train']
            test = data['test']
            pred_median = pd.Series(data['pred_median'], index=test.index)
            pred_samples = data['pred_samples']
            
            # 1. Tính MAE
            mae = np.mean(np.abs(pred_median - test))
            print(f"   🔹 Cột {col}: MAE = {mae:.4f}")
            
            summary_data.append({
                "Dataset": ds_name,
                "Model": MODEL_NAME_TAG,
                "Column": col,
                "MAE": mae,
                "Test_Points": len(test)
            })

            # 2. Vẽ biểu đồ
            plt.figure(figsize=(14, 7))
            
            # Vẽ 300 điểm lịch sử cuối (GPT-4o-mini nhớ dai hơn nên vẽ dài hơn chút)
            history_plot = train.iloc[-300:]
            
            plt.plot(history_plot.index, history_plot.values, label='History (Context)', color='gray', alpha=0.4)
            plt.plot(test.index, test.values, label='Ground Truth', color='black', linewidth=2)
            plt.plot(test.index, pred_median.values, label=f'GPT-4o-mini Prediction', color='#1f77b4', linestyle='--', linewidth=2)
            
            # Vẽ khoảng tin cậy 90%
            if pred_samples is not None:
                lower = np.quantile(pred_samples, 0.05, axis=0)
                upper = np.quantile(pred_samples, 0.95, axis=0)
                plt.fill_between(test.index, lower, upper, color='#1f77b4', alpha=0.15, label='Confidence Interval (90%)')

            plt.title(f"Forecast: {ds_name} - {col} | Model: {MODEL_NAME_TAG} | MAE: {mae:.2f}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Lưu ảnh
            img_path = os.path.join(img_dir, f"{col}_forecast.png")
            plt.savefig(img_path)
            plt.close()

    # Lưu báo cáo CSV
    if summary_data:
        report_path = os.path.join(OUTPUT_BASE_DIR, f"final_report_{MODEL_NAME_TAG}.csv")
        pd.DataFrame(summary_data).to_csv(report_path, index=False)
        print(f"\n✅ Đã lưu báo cáo tổng hợp tại: {report_path}")
        print(f"✅ Đã lưu biểu đồ tại các thư mục plots_{MODEL_NAME_TAG}")

if __name__ == "__main__":
    check_gpt4o_mini_results()