import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
import gc
from dotenv import load_dotenv

# --- CẤU HÌNH ---
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

# Dataset KPI
DATASETS_TO_RUN = {
    "networks_kpi": "network_traffic.csv"
}

MODEL_NAME = 'llama-3.2-3b' 

# [QUAN TRỌNG] Cấu hình lại để dự báo mượt mà hơn
llama_hypers = dict(
    temp=0.1,    # <--- GIẢM từ 0.7 xuống 0.1: Giúp đường dự báo ổn định, ít răng cưa
    alpha=0.90,  # <--- GIẢM từ 0.95 xuống 0.90: Bỏ qua các giá trị nhiễu đột biến
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=2, signed=True, half_bin_correction=True)
)

# --- HÀM LÀM MƯỢT DỮ LIỆU ---
def smooth_series(series, window_size=5):
    """
    Làm mượt dữ liệu bằng phương pháp Rolling Average.
    Với dữ liệu 30 phút/điểm, window_size=5 sẽ lấy trung bình trong khoảng 2.5 giờ.
    Điều này giúp loại bỏ các gai nhọn (noise) nhưng vẫn giữ được xu hướng chính.
    """
    # Rolling mean với center=True để không bị lệch pha thời gian
    smoothed = series.rolling(window=window_size, min_periods=1, center=True).mean()
    # Lấp đầy các giá trị NaN ở đầu/cuối chuỗi do rolling tạo ra
    smoothed = smoothed.ffill().bfill()
    return smoothed

def run_network_3b():
    print(f"ℹ️ Network KPI (Smoothed) | Model: {MODEL_NAME}")
    
    for ds_name, file_name in DATASETS_TO_RUN.items():
        print(f"\n" + "="*60)
        print(f"🚀 DATASET: {ds_name}")
        
        input_path = os.path.join(BASE_DIR, "datasets", ds_name, file_name)
        output_dir = os.path.join(BASE_DIR, f"output/{ds_name}_{MODEL_NAME}")
        output_file = os.path.join(output_dir, f"results_{ds_name}_{MODEL_NAME}.pkl")
        
        if not os.path.exists(input_path):
            print(f"❌ Thiếu data: {input_path}")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.read_csv(input_path)
        series = df['value']
        
        # Context: 30 phút/lần -> 48 điểm/ngày
        # limit_size 2000 ~ 41 ngày lịch sử
        limit_size = 2000 
        test_size = 48  # Dự báo 2 ngày tiếp theo
        
        if len(series) > limit_size:
            series = series.iloc[-limit_size:]
        
        train_raw = series.iloc[:-test_size]
        test = series.iloc[-test_size:]
        
        # [BƯỚC QUAN TRỌNG] LÀM MƯỢT DỮ LIỆU TRAIN TRƯỚC KHI ĐƯA VÀO MODEL
        print("   🧹 Đang làm mượt dữ liệu Train (Smoothing)...")
        train_smoothed = smooth_series(train_raw, window_size=5)

        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            print(f"   ⏳ Inference...")
            pred_dict = get_llmtime_predictions_data(
                train_smoothed, # <--- Đưa dữ liệu đã làm mượt vào model
                test, 
                model=MODEL_NAME, 
                num_samples=10, 
                **llama_hypers 
            )
            
            ds_results = {'value': {
                'train': train_raw, # Lưu lại train gốc để vẽ hình so sánh cho đúng thực tế
                'train_smoothed': train_smoothed, # Lưu thêm train smooth để debug
                'test': test,
                'pred_median': pred_dict['median'],
                'pred_samples': pred_dict['samples']
            }}
            
            with open(output_file, 'wb') as f:
                pickle.dump(ds_results, f)
            print(f"   ✅ Xong! Saved: {output_file}")
            
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_network_3b()