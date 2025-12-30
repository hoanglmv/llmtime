import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
import gc
from dotenv import load_dotenv

# --- 1. CẤU HÌNH ---
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ['OMP_NUM_THREADS'] = '4'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from huggingface_hub import login
    hf_token = os.getenv("HF_TOKEN")
    if hf_token: login(token=hf_token)
except: pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.serialize import SerializerSettings
from models.llmtime import get_llmtime_predictions_data

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- 2. DỮ LIỆU MỚI ---
# Dataset có biên độ biến thiên
DATASETS_TO_RUN = {
    "sin_wave_var": "sin_wave_var.csv" 
}

MODEL_NAME = 'llama-3.2-3b' 

llama_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=2, signed=True, half_bin_correction=True)
)

def run_sinwave_var():
    print(f"ℹ️ Đang chạy (Biên độ biến thiên) với Model: {MODEL_NAME}")
    
    for ds_name, file_name in DATASETS_TO_RUN.items():
        print(f"\n" + "="*60)
        print(f"🚀 DATASET: {ds_name}")
        print("="*60)
        
        # Input: datasets/sin_wave_var/sin_wave_var.csv
        input_path = os.path.join(BASE_DIR, "datasets", ds_name, file_name)
        
        # Output folder riêng biệt
        output_dir = os.path.join(BASE_DIR, f"output/{ds_name}_{MODEL_NAME}")
        output_file = os.path.join(output_dir, f"results_{ds_name}_{MODEL_NAME}.pkl")
        
        if not os.path.exists(input_path):
            print(f"❌ Chưa có data: {input_path}")
            print("   👉 Chạy util/create_sin1_dataset.py trước!")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.read_csv(input_path)
        series = df['value']
        
        print(f"   📊 Dữ liệu: {len(series)} dòng")
        
        # --- CẤU HÌNH CONTEXT ---
        # Vì tần số cao (chu kỳ hẹp), ta cần nhìn xa hơn một chút để thấy quy luật bao biên độ
        limit_size = 2000 
        test_size = 200 # Test dài hơn để xem model có vẽ tiếp được sóng không
        
        if len(series) > limit_size:
            series = series.iloc[-limit_size:]
        
        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]
        
        # Dọn dẹp RAM
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            print("   ⏳ Đang suy luận (Inference)...")
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

    print("\n🎉 HOÀN TẤT!")

if __name__ == "__main__":
    run_sinwave_var()
