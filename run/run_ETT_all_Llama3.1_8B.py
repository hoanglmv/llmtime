import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
import gc
from functools import partial
from dotenv import load_dotenv

# --- 1. CẤU HÌNH HỆ THỐNG ---
load_dotenv()
# Chọn GPU (0 hoặc 1 tùy server của bạn)
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ['OMP_NUM_THREADS'] = '4'
# Chống phân mảnh bộ nhớ GPU (Rất quan trọng)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from huggingface_hub import login
    hf_token = os.getenv("HF_TOKEN")
    if hf_token: login(token=hf_token)
except: pass

# Thêm đường dẫn project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import các module cốt lõi
from data.serialize import SerializerSettings
from models.llmtime import get_llmtime_predictions_data
from models.llms import completion_fns, context_lengths, tokenization_fns
from models.llama import llama_completion_fn, tokenize_fn as llama_tokenize_fn

# ==============================================================================
# 🛠️ CẤU HÌNH MODEL (RUNTIME INJECTION)
# ==============================================================================
# Tại đây chúng ta tự định nghĩa model Llama-3.1-8B để không phải sửa file gốc
# ------------------------------------------------------------------------------
# Tên model trên HuggingFace (Bạn nên dùng bản 8B này thay vì 3B vì nó thông minh hơn nhiều)
REAL_MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B"
# Key định danh nội bộ
MY_CUSTOM_KEY = "custom-llama-3.1-8b"

print(f"🛠️ Đang đăng ký model: {REAL_MODEL_PATH}...")

# 1. Đăng ký hàm dự đoán
completion_fns[MY_CUSTOM_KEY] = partial(llama_completion_fn, model=REAL_MODEL_PATH)

# 2. Đăng ký độ dài ngữ cảnh (Llama 3.1 hỗ trợ 128k, ta set 16k là quá đủ và nhẹ)
context_lengths[MY_CUSTOM_KEY] = 16000 

# 3. Đăng ký hàm tokenize
tokenization_fns[MY_CUSTOM_KEY] = partial(llama_tokenize_fn, model=REAL_MODEL_PATH)

print("✅ Đăng ký thành công!")

# ==============================================================================

# --- 2. CẤU HÌNH DỮ LIỆU ---
BASE_DIR = os.path.expanduser("/home/myvh07/hoanglmv/Project/llmtime")

DATASETS_TO_RUN = {
    "ETTm1": "ETTm1.csv",
    "ETTm2": "ETTm2.csv",
    "ETTh1": "ETTh1.csv",
    "ETTh2": "ETTh2.csv"
}

llama_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=2, signed=True, half_bin_correction=True)
)

# --- 3. HÀM LÀM SẠCH DỮ LIỆU (KHÔNG XÓA DÒNG) ---
def load_and_clean_data(file_path):
    print(f"   📖 Đang đọc và xử lý: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Xử lý Date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isna().sum() > 0:
            print(f"      ⚠️ Sửa lỗi ngày tháng (NaT) bằng ffill/bfill...")
            df['date'] = df['date'].ffill().bfill()
    
    # Xử lý Số liệu
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
def run_all_datasets():
    print(f"\nℹ️ Đang chạy với Model Key: {MY_CUSTOM_KEY}")
    
    for ds_name, file_name in DATASETS_TO_RUN.items():
        print(f"\n" + "#"*60)
        print(f"🚀 DATASET: {ds_name}")
        print("#"*60)
        
        input_path = os.path.join(BASE_DIR, "datasets/ETT-small", file_name)
        output_dir = os.path.join(BASE_DIR, f"output/{ds_name}")
        
        # Đổi tên file output để nhận diện model
        output_file = os.path.join(output_dir, f"results_{ds_name}_Llama3.1-8B.pkl")
        
        if not os.path.exists(input_path):
            print(f"❌ Không tìm thấy: {input_path}")
            continue
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dữ liệu
        df, target_cols = load_and_clean_data(input_path)
        ds_results = {}
        
        for col in target_cols:
            print(f"\n--- 🔄 {ds_name} | Cột: {col} ---")
            
            # Dọn dẹp GPU triệt để
            torch.cuda.empty_cache()
            gc.collect()

            series = df[col]
            
            # Cấu hình Context Window
            # Vì Llama 3.1 nhớ tốt, ta có thể tăng limit_size lên nếu muốn (ví dụ 3000)
            limit_size = 2000 
            test_size = 100
            
            if len(series) > limit_size:
                series = series.iloc[-limit_size:]
            
            train = series.iloc[:-test_size]
            test = series.iloc[-test_size:]
            
            try:
                pred_dict = get_llmtime_predictions_data(
                    train, test, 
                    model=MY_CUSTOM_KEY,   # <--- Dùng Key tự define ở trên
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
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()

        with open(output_file, 'wb') as f:
            pickle.dump(ds_results, f)
        print(f"\n💾 Đã lưu: {output_file}")

    print("\n🎉 HOÀN TẤT TOÀN BỘ!")

if __name__ == "__main__":
    run_all_datasets()