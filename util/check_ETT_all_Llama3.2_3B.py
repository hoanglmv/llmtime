import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
# Đường dẫn gốc project (Phải khớp với file run của bạn)
BASE_DIR = os.path.expanduser("/home/myvh07/hoanglmv/Project/llmtime")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "output")

# Danh sách dataset cần kiểm tra
# (Phải khớp với DATASETS_TO_RUN trong file run)
DATASETS = ["ETTm1", "ETTm2", "ETTh2"] 

# Hậu tố tên file kết quả. 
# Trong file run bạn code là: f"results_{ds_name}_Llama3B.pkl"
FILE_SUFFIX = "_Llama3B.pkl"

def check_results():
    print(f"📂 Đang kiểm tra kết quả tại thư mục: {OUTPUT_BASE_DIR}")
    
    # List chứa dữ liệu để xuất báo cáo CSV
    summary_data = []

    for ds_name in DATASETS:
        # Tái tạo đường dẫn file kết quả
        result_file = os.path.join(OUTPUT_BASE_DIR, ds_name, f"results_{ds_name}{FILE_SUFFIX}")
        
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
        img_dir = os.path.join(OUTPUT_BASE_DIR, ds_name, "plots_Llama3.2_3B")
        os.makedirs(img_dir, exist_ok=True)
        
        # Duyệt qua từng cột (HUFL, HULL, MUFL...)
        for col, data in all_results.items():
            train = data['train']
            test = data['test']
            pred_median = pd.Series(data['pred_median'], index=test.index)
            pred_samples = data['pred_samples'] # Dùng để vẽ khoảng tin cậy
            
            # --- 1. TÍNH TOÁN SAI SỐ (MAE) ---
            mae = np.mean(np.abs(pred_median - test))
            print(f"   🔹 Cột {col}: MAE = {mae:.4f}")
            
            # Lưu vào list tổng hợp
            summary_data.append({
                "Dataset": ds_name,
                "Model": "Llama-3.2-3B",
                "Column": col,
                "MAE": mae,
                "Test_Size": len(test)
            })

            # --- 2. VẼ BIỂU ĐỒ ---
            plt.figure(figsize=(14, 7))
            
            # Chỉ vẽ 200 điểm cuối của lịch sử (Context) để hình dễ nhìn
            history_plot = train.iloc[-200:]
            
            # Vẽ đường lịch sử (Context)
            plt.plot(history_plot.index, history_plot.values, label='History (Context)', color='gray', alpha=0.5)
            
            # Vẽ đường thực tế (Ground Truth)
            plt.plot(test.index, test.values, label='Actual (Ground Truth)', color='black', linewidth=2)
            
            # Vẽ đường dự báo (Prediction)
            plt.plot(test.index, pred_median.values, label='Llama 3.2 Prediction', color='#d62728', linestyle='--', linewidth=2)
            
            # Vẽ khoảng tin cậy 90% (Confidence Interval) nếu có samples
            if pred_samples is not None:
                # pred_samples có shape (num_samples, horizon) -> ví dụ (10, 100)
                lower = np.quantile(pred_samples, 0.05, axis=0)
                upper = np.quantile(pred_samples, 0.95, axis=0)
                plt.fill_between(test.index, lower, upper, color='#d62728', alpha=0.15, label='Confidence Interval (90%)')

            plt.title(f"Forecast: {ds_name} - Column {col}\nModel: Llama-3.2-3B | MAE: {mae:.2f}")
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
        report_path = os.path.join(OUTPUT_BASE_DIR, "final_report_Llama3.2_3B.csv")
        summary_df.to_csv(report_path, index=False)
        print(f"\n✅ Đã lưu báo cáo tổng hợp tại: {report_path}")
        print(f"✅ Đã lưu ảnh biểu đồ trong: {OUTPUT_BASE_DIR}/<Dataset>/plots_Llama3.2_3B/")
    else:
        print("\n❌ Không có dữ liệu nào được xử lý. Hãy kiểm tra lại file run đã chạy chưa.")

if __name__ == "__main__":
    check_results()