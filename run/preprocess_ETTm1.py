import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.expanduser("~/dttmai/hoanglmv/llmtime")
RAW_DATA_PATH = os.path.join(BASE_DIR, "datasets/ETT-small/ETTm1.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "datasets/ETT-small/ETTm1_cleaned.csv")
IMG_OUTPUT_PATH = os.path.join(BASE_DIR, "output/ETTm1/data_check.png")

# Tạo thư mục output nếu chưa có để lưu ảnh check
os.makedirs(os.path.dirname(IMG_OUTPUT_PATH), exist_ok=True)

def preprocess_data():
    print(f"🚀 Bắt đầu xử lý dữ liệu từ: {RAW_DATA_PATH}")
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Lỗi: Không tìm thấy file gốc tại {RAW_DATA_PATH}")
        return

    # 1. Đọc dữ liệu thô (low_memory=False để tránh cảnh báo mixed types ban đầu)
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    original_len = len(df)
    print(f"   📊 Tổng số dòng ban đầu: {original_len}")

    # 2. Xử lý cột Date
    # errors='coerce': Biến những dòng không phải ngày tháng (như dòng header lặp lại) thành NaT
    print("   🧹 Đang làm sạch cột Date...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Xóa các dòng bị NaT (chính là các dòng rác/header lặp)
    df = df.dropna(subset=['date'])
    print(f"   -> Đã xóa {original_len - len(df)} dòng rác (header lặp/lỗi format).")

    # 3. Xử lý các cột số
    target_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    print("   🔢 Đang chuẩn hóa các cột số liệu...")
    
    for col in target_cols:
        # Ép kiểu sang số thực (float), biến lỗi thành NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Xử lý NaN và trùng lặp
    before_dropna = len(df)
    df = df.dropna() # Xóa dòng có ô trống
    df = df.drop_duplicates(subset=['date']) # Xóa dòng trùng ngày giờ (nếu có)
    
    # Sắp xếp lại theo thời gian cho chuẩn Time Series
    df = df.sort_values(by='date').reset_index(drop=True)

    print(f"   -> Đã xóa thêm {before_dropna - len(df)} dòng chứa dữ liệu trống (NaN).")
    print(f"   ✅ Dữ liệu sạch cuối cùng: {len(df)} dòng.")

    # 5. Lưu file sạch
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"   💾 Đã lưu file sạch tại: {CLEAN_DATA_PATH}")

    # 6. Vẽ biểu đồ kiểm tra nhanh (Vẽ cột OT)
    plt.figure(figsize=(15, 5))
    plt.plot(df['date'], df['OT'], label='OT (Cleaned)', color='blue', linewidth=0.5)
    plt.title("Biểu đồ dữ liệu ETTm1 sau khi làm sạch (Cột OT)")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(IMG_OUTPUT_PATH)
    print(f"   🖼️  Đã lưu biểu đồ kiểm tra tại: {IMG_OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_data()
