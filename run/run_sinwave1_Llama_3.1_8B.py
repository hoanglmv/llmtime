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
# Model 8B nặng hơn, hãy đảm bảo chọn đúng GPU mạnh nhất bạn có (ví dụ '0' hoặc '1')
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
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

DATASETS_TO_RUN = {
    "sin_wave_var": "sin_wave_var.csv" 
}

# --- 3. CẤU HÌNH MODEL ---
MODEL_NAME = 'llama-3.1-8b' 

llama_hypers = dict(
    temp=0.7,
    alpha=0.99, # Tăng alpha lên 0.99 để xử lý biên độ tốt hơn
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=2, signed=True, half_bin_correction=True)
)

def run_sinwave_var_8b():
    print(f"ℹ️ Đang chạy (Biên độ biến thiên) với Model LỚN: {MODEL_NAME}")
    
    for ds_name, file_name in DATASETS_TO_RUN.items():
        print(f"\n" + "#"*60)
        print(f"🚀 DATASET: {ds_name}")
        print("#"*60)
        
        input_path = os.path.join(BASE_DIR, "datasets", ds_name, file_name)
        
        # Output folder riêng cho model 8B
        output_dir = os.path.join(BASE_DIR, f"output/{ds_name}_{MODEL_NAME}")
        output_file = os.path.join(output_dir, f"results_{ds_name}_{MODEL_NAME}.pkl")
        
        if not os.path.exists(input_path):
            print(f"❌ Không tìm thấy data: {input_path}")
            print("   👉 Hãy chạy util/create_sin1_dataset.py trước!")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"   📖 Đang đọc: {input_path}")
        df = pd.read_csv(input_path)
        series = df['value']
        
        # --- TỐI ƯU CONTEXT CHO BIẾN THIÊN ---
        # Tăng limit_size lên 3000 để model nhìn thấy chu kỳ biến thiên dài hơn
        limit_size = 3000 
        test_size = 200
        
        if len(series) > limit_size:
            series = series.iloc[-limit_size:]
        
        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]
        
        # Dọn dẹp RAM triệt để cho model 8B
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            print(f"   ⏳ Đang suy luận với Llama 3.1 8B (Context: {len(train)})...")
            pred_dict = get_llmtime_predictions_data(
                train, test, 
                model=MODEL_NAME, 
                num_samples=10, 
                **llama_hypers 
            )
            
            ds_results = {'value': {
                'train': train,
                'test': test,
                'pred_median': pred_dict['median'],
                'pred_samples': pred_dict['samples']
            }}
            
            with open(output_file, 'wb') as f:
                pickle.dump(ds_results, f)
            print(f"   ✅ Xong! Đã lưu tại: {output_file}")
            
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()

    print("\n🎉 HOÀN TẤT MODEL 8B!")

if __name__ == "__main__":
    run_sinwave_var_8b()
