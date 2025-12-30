import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
# Đường dẫn gốc project (Giống file run của bạn)
BASE_DIR = os.path.expanduser("/home/myvh07/hoanglmv/Project/llmtime")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "output")

# Danh sách dataset cần kiểm tra
DATASETS = ["ETTm1", "ETTm2", "ETTh1", "ETTh2"]

# Hậu tố tên file kết quả (Phải khớp với file run của bạn)
# File run của bạn lưu là: f"results_{ds_name}_Llama3.1-8B.pkl"
FILE_SUFFIX = "_Llama3.1-8B.pkl"

def check_llama_results():
    print(f"📂 Đang kiểm tra kết quả tại: {OUTPUT_BASE_DIR}")
    
    # DataFrame tổng hợp tất cả kết quả
    summary_data = []

    for ds_name in DATASETS:
        # Tái tạo lại đường dẫn file kết quả
        result_file = os.path.join(OUTPUT_BASE_DIR, ds_name, f"results_{ds_name}{FILE_SUFFIX}")
        
        # Kiểm tra file có tồn tại không
        if not os.path.exists(result_file):
            print(f"⚠️ Không tìm thấy kết quả cho {ds_name} (File: {result_file}). Bỏ qua.")
            continue
            
        print(f"\n" + "="*60)
        print(f"📊 ĐANG PHÂN TÍCH: {ds_name} (Llama-3.1-8B)")
        print("="*60)
        
        try:
            # Load dữ liệu từ file .pkl
            with open(result_file, 'rb') as f:
                all_results = pickle.load(f)
        except Exception as e:
            print(f"❌ Lỗi khi đọc file pickle: {e}")
            continue
        
        # Tạo thư mục lưu ảnh biểu đồ riêng cho Llama 3.1
        img_dir = os.path.join(OUTPUT_BASE_DIR, ds_name, "plots_Llama3.1_8B")
        os.makedirs(img_dir, exist_ok=True)
        
        # Duyệt qua từng cột (HUFL, HULL...)
        for col, data in all_results.items():
            train = data['train']
            test = data['test']
            pred_median = pd.Series(data['pred_median'], index=test.index)
            pred_samples = data['pred_samples'] # Dùng để vẽ khoảng tin cậy
            
            # 1. Tính sai số MAE (Mean Absolute Error)
            mae = np.mean(np.abs(pred_median - test))
            print(f"   🔹 Cột {col}: MAE = {mae:.4f}")
            
            # 2. Lưu vào list tổng hợp
            summary_data.append({
                "Dataset": ds_name,
                "Model": "Llama-3.1-8B",
                "Column": col,
                "MAE": mae,
                "Test_Points": len(test)
            })

            # 3. Vẽ biểu đồ so sánh
            plt.figure(figsize=(12, 6))
            
            # Chỉ vẽ 200 điểm cuối của lịch sử để hình dễ nhìn (Context gần nhất)
            history_plot = train.iloc[-200:]
            
            # Vẽ các đường
            plt.plot(history_plot.index, history_plot.values, label='History (Context)', color='gray', alpha=0.5)
            plt.plot(test.index, test.values, label='Ground Truth (Actual)', color='black', linewidth=2)
            plt.plot(test.index, pred_median.values, label='Llama-3.1 Prediction', color='blue', linestyle='--', linewidth=2)
            
            # Vẽ khoảng tin cậy 90% (từ sample thứ 5% đến 95%)
            if pred_samples is not None:
                # pred_samples thường có shape (num_samples, horizon) -> (10, 100)
                lower = np.quantile(pred_samples, 0.05, axis=0)
                upper = np.quantile(pred_samples, 0.95, axis=0)
                plt.fill_between(test.index, lower, upper, color='blue', alpha=0.15, label='Confidence Interval (90%)')

            plt.title(f"Llama 3.1 8B Forecast: {ds_name} - {col} (MAE: {mae:.2f})")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Lưu ảnh
            img_path = os.path.join(img_dir, f"{col}_Llama3.1.png")
            plt.savefig(img_path)
            plt.close()
            
    # 4. Lưu file CSV báo cáo tổng
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        report_path = os.path.join(OUTPUT_BASE_DIR, "final_report_Llama3.1_8B.csv")
        summary_df.to_csv(report_path, index=False)
        print(f"\n✅ Đã lưu báo cáo tổng hợp tại: {report_path}")
        print(f"✅ Đã lưu các biểu đồ so sánh trong thư mục: output/<Dataset>/plots_Llama3.1_8B/")
    else:
        print("\n❌ Không có dữ liệu nào được xử lý.")

if __name__ == "__main__":
    check_llama_results()