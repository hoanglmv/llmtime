import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(BASE_DIR, "output/ETTm1")
RESULT_FILE = os.path.join(OUTPUT_DIR, "results_ETTm1.pkl")

def evaluate_and_plot():
    print(f"📂 Đang load kết quả từ: {RESULT_FILE}")
    
    if not os.path.exists(RESULT_FILE):
        print("❌ Không tìm thấy file kết quả. Hãy chạy run/run_ETTm1.py trước!")
        return

    with open(RESULT_FILE, 'rb') as f:
        all_results = pickle.load(f)

    # DataFrame tổng hợp để lưu file CSV
    comparison_df = pd.DataFrame()

    for col, data in all_results.items():
        print(f"\n--- Đánh giá cột {col} ---")
        
        train = data['train']
        test = data['test']
        pred = pd.Series(data['pred_median'], index=test.index)
        
        # 1. Tính sai số MAE (Mean Absolute Error)
        mae = np.mean(np.abs(pred - test))
        print(f"   📉 MAE: {mae:.4f}")
        
        # 2. Lưu vào DataFrame so sánh
        col_compare = pd.DataFrame({
            f'{col}_ThucTe': test.values,
            f'{col}_DuBao': pred.values
        }, index=test.index)
        comparison_df = pd.concat([comparison_df, col_compare], axis=1)

        # 3. Vẽ biểu đồ
        plt.figure(figsize=(12, 6))
        
        # Vẽ 200 điểm cuối của train để thấy ngữ cảnh
        plt.plot(train.index[-200:], train.values[-200:], label='Lịch sử (Last 200)', color='gray', alpha=0.5)
        plt.plot(test.index, test.values, label='Thực tế', color='black', linewidth=2)
        plt.plot(test.index, pred.values, label='Dự báo (Llama)', color='purple', linestyle='--')
        
        # Vẽ khoảng tin cậy
        samples = data['pred_samples']
        lower = np.quantile(samples, 0.05, axis=0)
        upper = np.quantile(samples, 0.95, axis=0)
        plt.fill_between(test.index, lower, upper, alpha=0.3, color='purple', label='Độ tin cậy 90%')

        plt.title(f"Dự báo {col} trên tập dữ liệu ETTm1 (Llama-7B)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Lưu ảnh
        img_path = os.path.join(OUTPUT_DIR, f"chart_{col}.png")
        plt.savefig(img_path)
        print(f"   🖼️ Đã lưu biểu đồ: {img_path}")
        plt.close()

    # 4. Lưu file CSV tổng hợp (Như bạn yêu cầu)
    csv_path = os.path.join(OUTPUT_DIR, "final_prediction_compare.csv")
    comparison_df.to_csv(csv_path)
    print(f"\n✅ Đã lưu file CSV so sánh chi tiết tại: {csv_path}")
    print("Nội dung file CSV gồm các cột Thực tế vs Dự báo cho từng chỉ số.")

if __name__ == "__main__":
    evaluate_and_plot()
