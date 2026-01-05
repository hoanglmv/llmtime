import pandas as pd
import numpy as np
import os
import sys

# Xác định đường dẫn gốc
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets/sin_wave_var') # Folder mới

def create_dataset():
    # 1. Tạo thư mục
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # 2. Cấu hình sóng (Thử thách hơn)
    N = 4000 # Tăng số lượng điểm để model học được chu kỳ biến thiên biên độ
    t = np.linspace(0, 100, N)
    
    # --- LOGIC TẠO SÓNG ---
    # a. Sóng mang (Carrier): Tần số cao (Chu kỳ hẹp lại) -> sin(3*t)
    carrier_wave = np.sin(3 * t)
    
    # b. Đường bao biên độ (Envelope): Biên độ tự thay đổi theo sóng chậm -> sin(0.2*t)
    # Biên độ sẽ dao động trong khoảng
    amplitude_envelope = 700 + 200 * np.sin(0.2 * t) 
    
    # c. Tổng hợp: Nhân biên độ với sóng mang
    final_wave = amplitude_envelope * carrier_wave
    
    # Cộng thêm chút nhiễu và Trend nhẹ
    final_wave += np.random.normal(0, 10, N) # Nhiễu
    final_wave += t * 2 # Trend tăng nhẹ
    
    # 3. Tạo DataFrame
    date_range = pd.date_range(start='2024-01-01', periods=N, freq='H')
    
    df = pd.DataFrame({
        'date': date_range,
        'value': final_wave
    })
    
    # 4. Lưu file
    file_path = os.path.join(DATASET_DIR, 'sin_wave_var.csv')
    df.to_csv(file_path, index=False)
    
    print(f"✅ Đã tạo dataset biến thiên thành công!")
    print(f"   📂 Folder: {DATASET_DIR}")
    print(f"   📈 File: sin_wave_var.csv")
    print(f"   🌊 Đặc điểm: Tần số cao, Biên độ co giãn từ {df['value'].min():.0f} đến {df['value'].max():.0f}")

if __name__ == "__main__":
    create_dataset()
