import os
import sys
import pandas as pd
import numpy as np
import pickle
import time
from dotenv import load_dotenv

# --- 1. CẤU HÌNH HỆ THỐNG ---
load_dotenv()

# Kiểm tra API Key
if not os.getenv("OPENAI_API_KEY"):
    print("❌ LỖI: Chưa tìm thấy OPENAI_API_KEY trong file .env")
    sys.exit(1)

# Thêm đường dẫn project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.serialize import SerializerSettings
from models.llmtime import get_llmtime_predictions_data

# --- 2. CẤU HÌNH DỮ LIỆU ---
BASE_DIR = os.path.expanduser("/home/myvh07/hoanglmv/Project/llmtime")

DATASETS_TO_RUN = {
    "ETTm1": "ETTm1.csv",
    "ETTm2": "ETTm2.csv",
    "ETTh1": "ETTh1.csv",
    "ETTh2": "ETTh2.csv"
}

# --- CẤU HÌNH MODEL ---
# Sử dụng GPT-4o-mini (Model ngon-bổ-rẻ nhất hiện nay của OpenAI)
MODEL_NAME = "gpt-4o-mini"

gpt_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=2, signed=True, half_bin_correction=True)
)

# --- 3. HÀM LÀM SẠCH DỮ LIỆU ---
def load_and_clean_data(file_path):
    print(f"   📖 Đang đọc và xử lý: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Xử lý Date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isna().sum() > 0:
            print(f"      ⚠️ Sửa lỗi ngày tháng (NaT)...")
            df['date'] = df['date'].ffill().bfill()
    
    # Xử lý Số liệu (Thay NaN/0 bằng Epsilon)
    target_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    valid_cols = []
    EPSILON = 1e-5
    
    for col in target_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            c_nan = df[col].isna().sum()
            c_zero = (df[col] == 0).sum()
            
            if c_nan > 0: df[col] = df[col].fillna(EPSILON)
            if c_zero > 0: df[col] = df[col].replace(0, EPSILON)
                
            if c_nan > 0 or c_zero > 0:
                print(f"      🛠️  Cột '{col}': Sửa {c_nan} NaN và {c_zero} số 0.")
            
            valid_cols.append(col)
    
    if 'date' in df.columns:
        df = df.sort_values(by='date').reset_index(drop=True)
    
    print(f"   ✅ Dữ liệu sẵn sàng: {len(df)} dòng.")
    return df, valid_cols

# --- 4. HÀM CHẠY DỰ BÁO ---
def run_gpt4o_mini():
    print(f"\n" + "█"*60)
    print(f"🤖 ĐANG CHẠY MODEL: {MODEL_NAME}")
    print("█"*60)
        
    for ds_name, file_name in DATASETS_TO_RUN.items():
        print(f"\n👉 DATASET: {ds_name}")
        
        input_path = os.path.join(BASE_DIR, "datasets/ETT-small", file_name)
        output_dir = os.path.join(BASE_DIR, f"output/{ds_name}")
        
        # Tên file kết quả
        output_file = os.path.join(output_dir, f"results_{ds_name}_{MODEL_NAME}.pkl")
        
        if not os.path.exists(input_path):
            print(f"❌ Không tìm thấy file: {input_path}")
            continue
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dữ liệu
        df, target_cols = load_and_clean_data(input_path)
        ds_results = {}
        
        for col in target_cols:
            print(f"   ... Đang dự báo cột: {col}")
            
            series = df[col]
            
            # CẤU HÌNH INPUT
            # GPT-4o-mini có context window 128k, nên ta có thể lấy nhiều lịch sử hơn
            # GPT-3.5 cũ chỉ lấy được ~600, nhưng GPT-4o-mini lấy 2000-3000 vô tư
            limit_size = 2500 
            test_size = 100
            
            if len(series) > limit_size: series = series.iloc[-limit_size:]
            train = series.iloc[:-test_size]
            test = series.iloc[-test_size:]
            
            try:
                # Gọi API OpenAI
                pred_dict = get_llmtime_predictions_data(
                    train, test, 
                    model=MODEL_NAME,
                    num_samples=10,
                    **gpt_hypers 
                )
                
                ds_results[col] = {
                    'train': train,
                    'test': test,
                    'pred_median': pred_dict['median'],
                    'pred_samples': pred_dict['samples']
                }
                print(f"      ✅ Xong cột {col}")
                
                # Nghỉ 0.5s để tránh spam API quá nhanh
                time.sleep(0.5)

            except Exception as e:
                print(f"      ❌ Lỗi cột {col}: {e}")
                import traceback
                traceback.print_exc()

        # Lưu kết quả
        with open(output_file, 'wb') as f:
            pickle.dump(ds_results, f)
        print(f"💾 Đã lưu kết quả vào: {output_file}")

    print("\n🎉🎉🎉 HOÀN TẤT VỚI GPT-4o-Mini!")

if __name__ == "__main__":
    run_gpt4o_mini()