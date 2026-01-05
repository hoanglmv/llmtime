import pandas as pd
import numpy as np
import os
import sys

# Xác định đường dẫn gốc
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets/sin_wave')

def create_dataset():
    # 1. Tạo thư mục nếu chưa có
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # 2. Cấu hình sóng Sine (Biên độ lớn)
    N = 3000 # Số lượng điểm dữ liệu
    t = np.linspace(0, 100, N)
    
    # Tạo sóng tổng hợp: 
    # - Sóng chính: Amplitude = 1000, Frequency thấp
    # - Sóng nhiễu: Amplitude = 50, Frequency cao
    # - Trend: Cộng thêm t * 5 để dữ liệu đi lên nhẹ
    amplitude = 1000
    wave = amplitude * np.sin(t) + 50 * np.sin(5 * t) + (t * 5)
    
    # 3. Tạo DataFrame
    # Tạo cột ngày tháng giả định (tăng dần theo giờ)
    date_range = pd.date_range(start='2024-01-01', periods=N, freq='H')
    
    df = pd.DataFrame({
        'date': date_range,
        'value': wave  # Cột dữ liệu chính
    })
    
    # 4. Lưu file
    file_path = os.path.join(DATASET_DIR, 'sin_wave.csv')
    df.to_csv(file_path, index=False)
    
    print(f"✅ Đã tạo dataset thành công!")
    print(f"   📂 Đường dẫn: {file_path}")
    print(f"   🌊 Biên độ max: {df['value'].max():.2f}")
    print(f"   🌊 Biên độ min: {df['value'].min():.2f}")
    print(f"   📊 Số dòng: {len(df)}")

if __name__ == "__main__":
    create_dataset()
