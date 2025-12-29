import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
# Đường dẫn gốc project của bạn
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "output")

# Danh sách dataset cần kiểm tra
DATASETS = ["ETTm1", "ETTm2", "ETTh1", "ETTh2"]

# Danh sách model cần kiểm tra (phải khớp với tên trong file run)
# Lưu ý: Tên file đã được replace "/" thành "-"
MODELS = ["gpt-3.5-turbo-instruct"] 

def check_gpt_results():
    print(f"📂 Đang kiểm tra kết quả tại: {OUTPUT_BASE_DIR}")
    
    # DataFrame tổng hợp tất cả kết quả
    summary_df = pd.DataFrame()

    for ds_name in DATASETS:
        for model_name in MODELS:
            # Tái tạo lại tên file y hệt lúc lưu
            safe_model_name = model_name.replace("/", "-")
            result_file = os.path.join(OUTPUT_BASE_DIR, ds_name, f"results_{ds_name}_{safe_model_name}.pkl")
            
            # Kiểm tra file có tồn tại không
            if not os.path.exists(result_file):
                print(f"⚠️ Không tìm thấy kết quả cho {ds_name} - {model_name}. Bỏ qua.")
                continue
                
            print(f"\n" + "="*50)
            print(f"📊 ĐANG PHÂN TÍCH: {ds_name} ({model_name})")
            print("="*50)
            
            # Load dữ liệu từ file .pkl
            with open(result_file, 'rb') as f:
                all_results = pickle.load(f)
            
            # Thư mục lưu ảnh biểu đồ
            img_dir = os.path.join(OUTPUT_BASE_DIR, ds_name, "plots_gpt")
            os.makedirs(img_dir, exist_ok=True)
            
            # Duyệt qua từng cột (HUFL, HULL...)
            for col, data in all_results.items():
                train = data['train']
                test = data['test']
                pred_median = pd.Series(data['pred_median'], index=test.index)
                pred_samples = data['pred_samples'] # Dùng để vẽ khoảng tin cậy
                
                # 1. Tính sai số MAE
                mae = np.mean(np.abs(pred_median - test))
                print(f"   🔹 Cột {col}: MAE = {mae:.4f}")
                
                # 2. Lưu vào bảng tổng hợp
                row = {
                    "Dataset": ds_name,
                    "Model": model_name,
                    "Column": col,
                    "MAE": mae,
                    "Test_Points": len(test)
                }
                summary_df = pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)

                # 3. Vẽ biểu đồ so sánh
                plt.figure(figsize=(12, 6))
                
                # Chỉ vẽ 150 điểm cuối của lịch sử để hình dễ nhìn
                history_plot = train.iloc[-150:]
                
                plt.plot(history_plot.index, history_plot.values, label='History (Context)', color='gray', alpha=0.5)
                plt.plot(test.index, test.values, label='Ground Truth', color='black', linewidth=2)
                plt.plot(test.index, pred_median.values, label=f'GPT Prediction', color='green', linestyle='--')
                
                # Vẽ khoảng tin cậy 90% (từ sample thứ 5% đến 95%)
                if pred_samples is not None:
                    lower = np.quantile(pred_samples, 0.05, axis=0)
                    upper = np.quantile(pred_samples, 0.95, axis=0)
                    plt.fill_between(test.index, lower, upper, color='green', alpha=0.2, label='Confidence Interval (90%)')

                plt.title(f"GPT Forecast: {ds_name} - {col} (MAE: {mae:.2f})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Lưu ảnh
                img_path = os.path.join(img_dir, f"{col}_{safe_model_name}.png")
                plt.savefig(img_path)
                plt.close()

    # 4. Lưu file CSV báo cáo tổng
    if not summary_df.empty:
        report_path = os.path.join(OUTPUT_BASE_DIR, "final_gpt_report.csv")
        summary_df.to_csv(report_path, index=False)
        print(f"\n✅ Đã lưu báo cáo tổng hợp tại: {report_path}")
        print(f"✅ Đã lưu các biểu đồ so sánh trong thư mục: {OUTPUT_BASE_DIR}/<DatasetName>/plots_gpt/")
    else:
        print("\n❌ Không có dữ liệu nào được xử lý. Hãy kiểm tra lại file run_ETT_all_GPT.py đã chạy chưa.")

if __name__ == "__main__":
    check_gpt_results()