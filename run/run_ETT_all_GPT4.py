import os
import sys
import pandas as pd
import numpy as np
import pickle
import openai
import time
from dotenv import load_dotenv

# --- 1. CẤU HÌNH HỆ THỐNG ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    print("❌ LỖI: Chưa tìm thấy OPENAI_API_KEY trong file .env")
    sys.exit(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.serialize import SerializerSettings
from models.llmtime import get_llmtime_predictions_data

# --- 2. CẤU HÌNH DỮ LIỆU ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASETS_TO_RUN = {
    "ETTm1": "ETTm1.csv",
    # Mở comment các dòng dưới nếu muốn chạy tiếp, nhưng nhớ kiểm tra túi tiền
    # "ETTm2": "ETTm2.csv", 
}

# --- 3. CẤU HÌNH MODEL ---
MODEL_NAME = 'gpt-4' 

# Cấu hình "Tiết kiệm"
gpt_hypers = dict(
    temp=0.7,
    alpha=0.9, 
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=2, signed=True, half_bin_correction=True)
)

# --- 4. HÀM LÀM SẠCH DỮ LIỆU ---
def load_and_clean_data(file_path):
    print(f"   📖 Đang đọc và xử lý: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isna().sum() > 0: df['date'] = df['date'].ffill().bfill()
    
    target_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    valid_cols = []
    EPSILON = 1e-5 
    for col in target_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().sum() > 0: df[col] = df[col].fillna(EPSILON)
            if (df[col] == 0).sum() > 0: df[col] = df[col].replace(0, EPSILON)
            valid_cols.append(col)
    if 'date' in df.columns: df = df.sort_values(by='date').reset_index(drop=True)
    return df, valid_cols

# --- 5. HÀM CHẠY DỰ BÁO (TIẾT KIỆM & AN TOÀN) ---
def run_gpt_datasets():
    print(f"ℹ️ Đang chạy với Model OpenAI: {MODEL_NAME}")
    print("💰 CHẾ ĐỘ TIẾT KIỆM: Limit=1500, Samples=3")

    for ds_name, file_name in DATASETS_TO_RUN.items():
        print(f"\n" + "#"*60)
        print(f"🚀 DATASET: {ds_name}")
        print("#"*60)
        
        input_path = os.path.join(BASE_DIR, "datasets/ETT-small", file_name)
        output_dir = os.path.join(BASE_DIR, f"output/{ds_name}")
        output_file = os.path.join(output_dir, f"results_{ds_name}_{MODEL_NAME}.pkl")
        
        if not os.path.exists(input_path): continue
        os.makedirs(output_dir, exist_ok=True)
        
        df, target_cols = load_and_clean_data(input_path)
        
        # Resume Mode: Load kết quả cũ để không chạy lại tốn tiền
        if os.path.exists(output_file):
            try:
                with open(output_file, 'rb') as f:
                    ds_results = pickle.load(f)
                print(f"   📂 Đã load {len(ds_results)} cột từ file cũ.")
            except: ds_results = {}
        else:
            ds_results = {}
        
        for col in target_cols:
            if col in ds_results:
                print(f"   ⏩ Cột {col} đã có kết quả. Bỏ qua.")
                continue

            print(f"\n--- 🔄 {ds_name} | Cột: {col} ---")
            series = df[col]
            
            # [QUAN TRỌNG] CẤU HÌNH TIẾT KIỆM
            # 1500 điểm = khoảng 2 tuần dữ liệu (ETTm1), đủ để bắt chu kì
            # Đừng tăng lên 3000 nếu muốn tiết kiệm
            limit_size = 1500 
            test_size = 100
            
            if len(series) > limit_size: series = series.iloc[-limit_size:]
            train, test = series.iloc[:-test_size], series.iloc[-test_size:]
            
            # --- RETRY LOOP ---
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    pred_dict = get_llmtime_predictions_data(
                        train, test, 
                        model=MODEL_NAME, 
                        # [QUAN TRỌNG] Giảm samples xuống 3. 
                        # Mặc định là 10-20. Số 3 là tối thiểu để lấy trung vị (median)
                        num_samples=3, 
                        **gpt_hypers 
                    )
                    
                    ds_results[col] = {
                        'train': train,
                        'test': test,
                        'pred_median': pred_dict['median'],
                        'pred_samples': pred_dict['samples']
                    }
                    print(f"   ✅ Xong cột {col}")
                    
                    # Lưu ngay lập tức
                    with open(output_file, 'wb') as f:
                        pickle.dump(ds_results, f)
                    
                    # [QUAN TRỌNG] Ngủ 60s để reset Rate Limit về 0
                    print("   💤 Nghỉ 60s để bảo vệ ví tiền và Rate Limit...")
                    time.sleep(60)
                    break 

                except Exception as e:
                    err_msg = str(e)
                    if "Rate limit" in err_msg:
                        wait_time = 90
                        print(f"   ⚠️ Rate Limit! (Thử {attempt+1}/{max_retries}). Đợi {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"   ❌ Lỗi cột {col}: {e}")
                        # Lỗi khác (không phải rate limit) thì dừng để kiểm tra code
                        break 

    print("\n🎉 HOÀN TẤT TOÀN BỘ!")

if __name__ == "__main__":
    run_gpt_datasets()