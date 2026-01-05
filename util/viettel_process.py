import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import sys

# --- CẤU HÌNH SERVER (QUAN TRỌNG) ---
# Sử dụng backend 'Agg' để vẽ hình mà không cần màn hình (tránh lỗi trên remote server)
import matplotlib
matplotlib.use('Agg')

# --- CẤU HÌNH NGƯỜI DÙNG ---
TARGET_CELL = 'enodebB9'           # Tên trạm cần xử lý
START_DATE_WEEK = '2025-11-01'     # Ngày bắt đầu vẽ biểu đồ tuần
INPUT_FILE = 'viettel.csv'         # Tên file đầu vào
OUTPUT_IMG_NAME = f'chart_{TARGET_CELL}_{START_DATE_WEEK}.png'
OUTPUT_CSV_NAME = f'{TARGET_CELL}.csv'

# Định nghĩa đường dẫn (Tự động tính toán dựa trên vị trí file script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '../datasets/viettel') # Folder chứa data
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '../output')         # Folder chứa kết quả (ảnh + csv)

# Tạo folder output nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------------------

def extract_minutes(time_str):
    """Hàm xử lý chuỗi phút giây: '45:00.0' -> lấy 45"""
    try:
        return int(str(time_str).split(':')[0])
    except:
        return 0

def load_and_process_data():
    """Đọc file, xử lý thời gian và trả về DataFrame sạch"""
    file_path = os.path.join(DATA_DIR, INPUT_FILE)
    
    print(f"[1/4] Đang đọc dữ liệu từ: {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file tại {file_path}")
        sys.exit(1)

    df = pd.read_csv(file_path)
    print(f"   -> Đã load {len(df)} dòng.")

    print("[2/4] Đang xử lý cột thời gian (Time Mapping)...")
    # 1. Date Hour
    df['base_time'] = pd.to_datetime(df['date_hour'], format='%Y-%m-%d-%H')
    
    # 2. Minute Offset
    df['minute_offset'] = df['update_time'].apply(extract_minutes)
    
    # 3. Final Timestamp
    df['timestamp'] = df['base_time'] + pd.to_timedelta(df['minute_offset'], unit='m')
    
    # 4. Cleanup
    df_clean = df.drop(columns=['base_time', 'minute_offset'])
    df_clean = df_clean.sort_values(by=['cell_name', 'timestamp'])
    
    return df_clean

def export_cell_data(df, cell_name):
    """Lọc dữ liệu 1 trạm và lưu ra CSV"""
    print(f"[3/4] Đang xuất dữ liệu trạm {cell_name} ra CSV...")
    
    df_cell = df[df['cell_name'] == cell_name].copy()
    
    if df_cell.empty:
        print(f"❌ Cảnh báo: Không tìm thấy trạm {cell_name}")
        return None

    # Chọn các cột quan trọng để file nhẹ hơn
    cols_to_keep = ['timestamp', 'ps_traffic_mb', 'avg_rrc_connected_user', 'prb_dl_used']
    # Chỉ giữ lại các cột có thực
    cols_to_keep = [c for c in cols_to_keep if c in df_cell.columns]
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
    df_cell[cols_to_keep].to_csv(output_path, index=False)
    
    print(f"   -> Đã lưu: {output_path}")
    return df_cell

def visualize_week(df_cell, start_date_str):
    """Vẽ biểu đồ 1 tuần và LƯU ẢNH"""
    print(f"[4/4] Đang vẽ và lưu biểu đồ tuần từ ngày {start_date_str}...")
    
    if df_cell is None or df_cell.empty:
        print("   -> Không có dữ liệu để vẽ.")
        return

    # Lọc tuần
    start_date = pd.to_datetime(start_date_str)
    end_date = start_date + pd.Timedelta(days=7)
    
    df_plot = df_cell[
        (df_cell['timestamp'] >= start_date) & 
        (df_cell['timestamp'] < end_date)
    ].sort_values('timestamp')
    
    if df_plot.empty:
        print(f"❌ Không có dữ liệu trong khoảng {start_date_str} - {end_date}")
        return

    # Cấu hình vẽ
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

    # 1. Traffic
    sns.lineplot(ax=axes[0], data=df_plot, x='timestamp', y='ps_traffic_mb', color='tab:blue', marker='o', label='Traffic')
    axes[0].set_title(f'Traffic (MB) - {TARGET_CELL} (Tuần {start_date_str})', fontweight='bold')
    axes[0].set_ylabel('MB')
    axes[0].legend(loc='upper right')

    # 2. User
    sns.lineplot(ax=axes[1], data=df_plot, x='timestamp', y='avg_rrc_connected_user', color='tab:orange', marker='o', label='Users')
    axes[1].set_ylabel('Users')

    # 3. PRB
    sns.lineplot(ax=axes[2], data=df_plot, x='timestamp', y='prb_dl_used', color='tab:green', marker='o', label='PRB Used')
    axes[2].set_ylabel('PRB Used')
    axes[2].set_xlabel('Thời gian')

    # Format trục thời gian
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %Hh'))
    axes[2].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[2].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # LƯU ẢNH
    output_img_path = os.path.join(OUTPUT_DIR, OUTPUT_IMG_NAME)
    plt.savefig(output_img_path)
    print(f"✅ ĐÃ LƯU ẢNH BIỂU ĐỒ TẠI: {output_img_path}")
    
    # Dọn dẹp bộ nhớ
    plt.close()

# --- MAIN ---
if __name__ == "__main__":
    # Chạy quy trình
    df_clean = load_and_process_data()
    df_cell = export_cell_data(df_clean, TARGET_CELL)
    visualize_week(df_cell, START_DATE_WEEK)
    print("\n=== HOÀN TẤT ===")