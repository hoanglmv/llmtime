import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
import gc
from dotenv import load_dotenv

# --- 1. CẤU HÌNH HỆ THỐNG ---
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['OMP_NUM_THREADS'] = '4'

try:
    from huggingface_hub import login
    hf_token = os.getenv("HF_TOKEN")
    if hf_token: login(token=hf_token)
except: pass

# Thêm đường dẫn để import module của project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.serialize import SerializerSettings
from models.llmtime import get_llmtime_predictions_data

# --- 2. CẤU HÌNH DỮ LIỆU ---
BASE_DIR = os.path.expanduser("/home/myvh07/hoanglmv/Project/llmtime")

# Danh sách các dataset muốn chạy
DATASETS_TO_RUN = {
    
    "ETTm1": "ETTm1.csv",
    "ETTm2": "ETTm2.csv",
    "ETTh2": "ETTh2.csv",
    "ETTh1": "ETTh1.csv"
}

# Cấu hình Model Llama-7B
llama_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=2, signed=True, half_bin_correction=True)
)

# --- 3. HÀM LÀM SẠCH DỮ LIỆU (KHÔNG XÓA DÒNG) ---
def load_and_clean_data(file_path):
    """
    Đọc file. Tuyệt đối KHÔNG xóa dòng nào.
    - Date lỗi (NaT) -> Điền bằng ngày trước đó (ffill).
    - Giá trị lỗi (NaN/0) -> Điền bằng epsilon.
    """
    print(f"   📖 Đang đọc và xử lý: {file_path}")
    
    # low_memory=False để đọc hết file vào RAM
    df = pd.read_csv(file_path, low_memory=False)
    
    # 1. Xử lý cột date
    if 'date' in df.columns:
        # Chuyển đổi sang datetime, lỗi thành NaT
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Đếm lỗi
        n_date_err = df['date'].isna().sum()
        if n_date_err > 0:
            print(f"      ⚠️ Có {n_date_err} dòng lỗi ngày tháng (NaT). Đang tự động điền (ffill)...")
            # Fill ngày tháng bằng giá trị của dòng trước đó để không phải xóa dòng
            df['date'] = df['date'].ffill()
            # Nếu dòng đầu tiên bị NaT thì dùng backfill
            df['date'] = df['date'].bfill()
    
    # 2. Xử lý các cột số liệu
    target_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    valid_cols = []
    EPSILON = 1e-5  # Giá trị nhỏ thay thế cho 0 và NaN
    
    for col in target_cols:
        if col in df.columns:
            # Ép kiểu số, biến lỗi (như chữ text) thành NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # --- LOGIC MỚI: FILL TOÀN BỘ, KHÔNG XÓA ---
            # Đếm NaN và 0 để báo cáo
            count_nan = df[col].isna().sum()
            count_zero = (df[col] == 0).sum()
            
            # Thay thế NaN bằng Epsilon
            if count_nan > 0:
                df[col] = df[col].fillna(EPSILON)
            
            # Thay thế 0 bằng Epsilon
            if count_zero > 0:
                df[col] = df[col].replace(0, EPSILON)
                
            if count_nan > 0 or count_zero > 0:
                print(f"      🛠️  Cột '{col}': Đã thay thế {count_nan} ô NaN và {count_zero} ô số 0 bằng {EPSILON}")
            
            valid_cols.append(col)
    
    # 3. Sort lại theo thời gian
    # Lưu ý: Vì đã fill hết NaT nên sort sẽ ổn định
    if 'date' in df.columns:
        df = df.sort_values(by='date').reset_index(drop=True)
    
    print(f"   ✅ Dữ liệu sẵn sàng: {len(df)} dòng. Các cột hợp lệ: {valid_cols}")
    return df, valid_cols

# --- 4. HÀM CHẠY DỰ BÁO ---
def run_all_datasets():
    for ds_name, file_name in DATASETS_TO_RUN.items():
        print(f"\n" + "#"*60)
        print(f"🚀 BẮT ĐẦU XỬ LÝ DATASET: {ds_name}")
        print("#"*60)
        
        # Đường dẫn file
        input_path = os.path.join(BASE_DIR, "datasets/ETT-small", file_name)
        output_dir = os.path.join(BASE_DIR, f"output/{ds_name}")
        output_file = os.path.join(output_dir, f"results_{ds_name}.pkl")
        
        if not os.path.exists(input_path):
            print(f"❌ Không tìm thấy file: {input_path}. Bỏ qua.")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dữ liệu (Hàm mới không xóa dòng)
        df, target_cols = load_and_clean_data(input_path)
        
        # Dictionary lưu kết quả của dataset này
        ds_results = {}
        
        # Chạy từng cột
        for col in target_cols:
            print(f"\n--- 🔄 {ds_name} | Cột: {col} ---")
            
            # --- DỌN DẸP MEMORY ---
            torch.cuda.empty_cache()
            gc.collect()

            series = df[col]
            
            # Cấu hình split train/test (Lấy 2000 dòng cuối)
            limit_size = 2000 
            test_size = 100
            
            if len(series) > limit_size:
                series = series.iloc[-limit_size:]
            
            train = series.iloc[:-test_size]
            test = series.iloc[-test_size:]
            
            try:
                # Gọi Model
                pred_dict = get_llmtime_predictions_data(
                    train, test, 
                    model='llama-7b',
                    num_samples=10,
                    **llama_hypers 
                )
                
                ds_results[col] = {
                    'train': train,
                    'test': test,
                    'pred_median': pred_dict['median'],
                    'pred_samples': pred_dict['samples']
                }
                print(f"   ✅ Xong cột {col}")

            except Exception as e:
                print(f"   ❌ Lỗi cột {col}: {e}")
                torch.cuda.empty_cache()

        # Lưu kết quả
        with open(output_file, 'wb') as f:
            pickle.dump(ds_results, f)
        print(f"\n💾 Đã lưu kết quả {ds_name} vào: {output_file}")

    print("\n🎉🎉🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH! 🎉🎉🎉")

if __name__ == "__main__":
    run_all_datasets()