import pandas as pd
import numpy as np
import os
import sys

# Xác định đường dẫn gốc
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets/networks_kpi')

def create_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Cấu hình thời gian: 30 phút/lần trong 120 ngày (đủ dài để model học)
    start_date = '2024-01-01'
    periods = 48 * 120 # 48 điểm/ngày * 120 ngày
    freq = '30min'
    
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # --- LOGIC TẠO KPI MẠNG ---
    # Định nghĩa các mốc giờ và mức độ tải (0-100%)
    # Giờ: [0, 5, 8, 12, 14, 20, 22, 24]
    # Tải: [Thấp, Thấp, Cao, Giảm trưa, Cao chiều, Đỉnh tối, Giảm nhẹ, Thấp]
    key_hours = [0, 5, 8, 12, 14, 20, 22, 24]
    key_values = [10, 15, 85, 60, 85, 95, 80, 10]
    
    traffic_data = []
    
    for dt in date_range:
        # Lấy giờ hiện tại dưới dạng thập phân (VD: 14:30 -> 14.5)
        current_hour = dt.hour + dt.minute / 60.0
        
        # Nội suy tuyến tính để tạo đường cong mượt giữa các mốc
        base_load = np.interp(current_hour, key_hours, key_values)
        
        # Thêm Noise (Nhiễu ngẫu nhiên)
        noise = np.random.normal(0, 5) 
        
        # Thêm biến động theo ngày cuối tuần (Cuối tuần thường cao hơn chút hoặc khác biệt)
        if dt.dayofweek >= 5: # Thứ 7, CN
            base_load *= 1.1 # Tăng 10% traffic giải trí
            
        final_value = max(0, base_load + noise)
        traffic_data.append(final_value)
        
    df = pd.DataFrame({
        'date': date_range,
        'value': traffic_data
    })
    
    file_path = os.path.join(DATASET_DIR, 'network_traffic.csv')
    df.to_csv(file_path, index=False)
    
    print(f"✅ Đã tạo dataset Network KPI thành công!")
    print(f"   📂 Folder: {DATASET_DIR}")
    print(f"   📊 Số dòng: {len(df)}")
    print(f"   📈 Max Load: {df['value'].max():.2f}")

if __name__ == "__main__":
    create_dataset()
