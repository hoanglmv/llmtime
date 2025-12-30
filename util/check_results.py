import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CẤU HÌNH KIỂM TRA ---
# Đường dẫn gốc project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "output")

# [QUAN TRỌNG] Nhập tên Model bạn muốn kiểm tra tại đây (phải khớp với MODEL_NAME trong file run)
# Ví dụ: 'llama-3.2-3b' hoặc 'llama-3.1-8b'
MODEL_TO_CHECK = 'llama-3.1-8b' 

# Danh sách dataset cần kiểm tra (đã thêm ETTh1)
DATASETS = ["ETTm1", "ETTm2", "ETTh1", "ETTh2"]

def check_results():
    print(f"📂 Đang kiểm tra kết quả tại thư mục: {OUTPUT_BASE_DIR}")
    print(f"🤖 Model mục tiêu: {MODEL_TO_CHECK}")
    
    # List chứa dữ liệu để xuất báo cáo CSV
    summary_data = []

    for ds_name in DATASETS:
        # Tái tạo đường dẫn file kết quả theo quy ước mới
        file_name = f"results_{ds_name}_{MODEL_TO_CHECK}.pkl"
        result_file = os.path.join(OUTPUT_BASE_DIR, ds_name, file_name)
        
        # Kiểm tra file có tồn tại không
        if not os.path.exists(result_file):
            print(f"⚠️ Không tìm thấy kết quả cho {ds_name} (File: {result_file}). Bỏ qua.")
            continue
            
        print(f"\n" + "="*60)
        print(f"📊 ĐANG PHÂN TÍCH DATASET: {ds_name}")
        print("="*60)
        
        try:
            # Load dữ liệu từ file .pkl
            with open(result_file, 'rb') as f:
                all_results = pickle.load(f)
        except Exception as e:
            print(f"❌ Lỗi khi đọc file pickle: {e}")
            continue
        
        # Tạo thư mục để lưu ảnh biểu đồ
        plot_folder_name = f"plots_{MODEL_TO_CHECK}"
        img_dir = os.path.join(OUTPUT_BASE_DIR, ds_name, plot_folder_name)
        os.makedirs(img_dir, exist_ok=True)
        
        # Duyệt qua từng cột (HUFL, HULL, MUFL...)
        for col, data in all_results.items():
            train = data['train']
            test = data['test']
            
            # Xử lý dữ liệu dự báo
            # pred_median có thể là Series hoặc array
            if isinstance(data['pred_median'], pd.Series):
                pred_median = data['pred_median']
            else:
                pred_median = pd.Series(data['pred_median'], index=test.index)
                
            pred_samples = data['pred_samples'] # Dùng để vẽ khoảng tin cậy
            
            # --- 1. TÍNH TOÁN SAI SỐ (MAE & MSE) ---
            mae = np.mean(np.abs(pred_median - test))
            mse = np.mean((pred_median - test) ** 2)
            
            print(f"   🔹 Cột {col}: MAE = {mae:.4f}")
            
            # Lưu vào list tổng hợp
            summary_data.append({
                "Dataset": ds_name,
                "Model": MODEL_TO_CHECK,
                "Column": col,
                "MAE": mae,
                "MSE": mse,
                "Test_Size": len(test)
            })

            # --- 2. VẼ BIỂU ĐỒ ---
            plt.figure(figsize=(14, 7))
            
            # Chỉ vẽ 150 điểm cuối của lịch sử (Context) để hình dễ nhìn
            history_plot = train.iloc[-150:]
            
            # Vẽ đường lịch sử (Context)
            plt.plot(history_plot.index, history_plot.values, label='History (Context)', color='gray', alpha=0.5)
            
            # Vẽ đường thực tế (Ground Truth)
            plt.plot(test.index, test.values, label='Actual (Ground Truth)', color='black', linewidth=2)
            
            # Vẽ đường dự báo (Prediction)
            plt.plot(test.index, pred_median.values, label=f'{MODEL_TO_CHECK} Prediction', color='#d62728', linestyle='--', linewidth=2)
            
            # Vẽ khoảng tin cậy 90% (Confidence Interval) nếu có samples
            if pred_samples is not None:
                # pred_samples thường là DataFrame (samples x time)
                # Tính quantile theo cột (axis=0)
                try:
                    lower = np.quantile(pred_samples, 0.05, axis=0)
                    upper = np.quantile(pred_samples, 0.95, axis=0)
                    plt.fill_between(test.index, lower, upper, color='#d62728', alpha=0.15, label='Confidence Interval (90%)')
                except Exception as e:
                    print(f"      ⚠️ Không vẽ được Confidence Interval: {e}")

            plt.title(f"Forecast: {ds_name} - Column {col}\nModel: {MODEL_TO_CHECK} | MAE: {mae:.4f}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Lưu ảnh
            img_path = os.path.join(img_dir, f"{col}_forecast.png")
            plt.savefig(img_path)
            plt.close() # Đóng plot để giải phóng RAM
            
    # --- 3. LƯU BÁO CÁO TỔNG HỢP ---
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        # Tính trung bình MAE toàn bộ
        avg_mae = summary_df['MAE'].mean()
        print(f"\n🌟 TRUNG BÌNH MAE TOÀN BỘ DATASETS: {avg_mae:.4f}")
        
        report_path = os.path.join(OUTPUT_BASE_DIR, f"final_report_{MODEL_TO_CHECK}.csv")
        summary_df.to_csv(report_path, index=False)
        print(f"✅ Đã lưu báo cáo tổng hợp tại: {report_path}")
        print(f"✅ Đã lưu ảnh biểu đồ trong: {OUTPUT_BASE_DIR}/<Dataset>/plots_{MODEL_TO_CHECK}/")
    else:
        print("\n❌ Không có dữ liệu nào được xử lý. Hãy kiểm tra lại:\n1. Tên Model trong file check có khớp file run không?\n2. Đã chạy file run thành công chưa?")

if __name__ == "__main__":
    check_results()