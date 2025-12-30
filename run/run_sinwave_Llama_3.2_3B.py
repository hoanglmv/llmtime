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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from huggingface_hub import login
    hf_token = os.getenv("HF_TOKEN")
    if hf_token: login(token=hf_token)
except: pass

# Thêm đường dẫn project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.serialize import SerializerSettings
from models.llmtime import get_llmtime_predictions_data

# --- 2. CẤU HÌNH DỮ LIỆU ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Tên Folder: Tên File CSV
DATASETS_TO_RUN = {
    "sin_wave": "sin_wave.csv" 
}

# --- 3. CẤU HÌNH MODEL ---
MODEL_NAME = 'llama-3.2-3b' 

llama_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=2, signed=True, half_bin_correction=True)
)

def run_sinwave():
    print(f"ℹ️ Đang chạy với Model Key: {MODEL_NAME}")
    
    for ds_name, file_name in DATASETS_TO_RUN.items():
        print(f"\n" + "#"*60)
        print(f"🚀 DATASET: {ds_name}")
        print("#"*60)
        
        # Đường dẫn: datasets/sin_wave/sin_wave.csv
        input_path = os.path.join(BASE_DIR, "datasets", ds_name, file_name)
        
        # Output: output/sin_wave_llama-3.2-3b/
        output_dir = os.path.join(BASE_DIR, f"output/{ds_name}_{MODEL_NAME}")
        output_file = os.path.join(output_dir, f"results_{ds_name}_{MODEL_NAME}.pkl")
        
        if not os.path.exists(input_path):
            print(f"❌ Không tìm thấy file data: {input_path}")
            print("   👉 Hãy chạy file util/create_sin_dataset.py trước!")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Đọc dữ liệu
        print(f"   📖 Đang đọc: {input_path}")
        df = pd.read_csv(input_path)
        
        # Chỉ chạy cột 'value'
        target_cols = ['value']
        
        ds_results = {}
        
        for col in target_cols:
            if col not in df.columns: continue
            
            print(f"\n--- 🔄 {ds_name} | Cột: {col} ---")
            
            # Dọn dẹp RAM trước khi chạy
            torch.cuda.empty_cache()
            gc.collect()

            series = df[col]
            
            # Cấu hình: Lấy 1000 điểm cuối để dự báo 100 điểm tiếp theo
            limit_size = 1000 
            test_size = 100
            
            if len(series) > limit_size:
                series = series.iloc[-limit_size:]
            
            train = series.iloc[:-test_size]
            test = series.iloc[-test_size:]
            
            try:
                pred_dict = get_llmtime_predictions_data(
                    train, test, 
                    model=MODEL_NAME, 
                    num_samples=10, # Lấy 10 mẫu để vẽ vùng tin cậy
                    **llama_hypers 
                )
                
                ds_results[col] = {
                    'train': train,
                    'test': test,
                    'pred_median': pred_dict['median'],
                    'pred_samples': pred_dict['samples']
                }
                print(f"   ✅ Xong cột {col}")

                # Xóa biến tạm để giải phóng RAM ngay
                del pred_dict

            except Exception as e:
                print(f"   ❌ Lỗi cột {col}: {e}")
                import traceback
                traceback.print_exc()

        with open(output_file, 'wb') as f:
            pickle.dump(ds_results, f)
        print(f"\n💾 Đã lưu kết quả: {output_file}")

    print("\n🎉 HOÀN TẤT!")

if __name__ == "__main__":
    run_sinwave()
