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
    print("👉 Vui lòng thêm dòng: OPENAI_API_KEY=sk-proj-...")
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

# --- 3. CẤU HÌNH MODEL GPT ---
# Lưu ý: "text-davinci-003" (GPT-3) đã ngừng hoạt động.
# Dưới đây là danh sách các model thay thế:
MODELS_CONFIG = [
    # Model thay thế cho GPT-3 (Completion style) - Tốt nhất cho LLMTime
    "gpt-3.5-turbo-instruct", 
    
    # Model GPT-3.5 Chat (Có thể dùng nhưng format prompt của llmtime tối ưu cho instruct hơn)
    # "gpt-3.5-turbo", 
]

gpt_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=2, signed=True, half_bin_correction=True)
)

# --- 4. HÀM LÀM SẠCH DỮ LIỆU (GIỮ NGUYÊN) ---
def load_and_clean_data(file_path):
    """
    Đọc file. Tuyệt đối KHÔNG xóa dòng nào.
    - Date lỗi (NaT) -> Điền bằng ngày trước đó (ffill).
    - Giá trị lỗi (NaN/0) -> Điền bằng epsilon.
    """
    print(f"   📖 Đang đọc và xử lý: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        n_date_err = df['date'].isna().sum()
        if n_date_err > 0:
            print(f"      ⚠️ Có {n_date_err} dòng lỗi ngày tháng. Auto-fill...")
            df['date'] = df['date'].ffill().bfill()
    
    target_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    valid_cols = []
    EPSILON = 1e-5
    
    for col in target_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            count_nan = df[col].isna().sum()
            count_zero = (df[col] == 0).sum()
            
            if count_nan > 0: df[col] = df[col].fillna(EPSILON)
            if count_zero > 0: df[col] = df[col].replace(0, EPSILON)
                
            if count_nan > 0 or count_zero > 0:
                print(f"      🛠️  Cột '{col}': Sửa {count_nan} NaN và {count_zero} số 0.")
            valid_cols.append(col)
    
    if 'date' in df.columns:
        df = df.sort_values(by='date').reset_index(drop=True)
    
    print(f"   ✅ Dữ liệu sẵn sàng: {len(df)} dòng.")
    return df, valid_cols

# --- 5. HÀM CHẠY DỰ BÁO ---
def run_all_datasets_gpt():
    
    # Vòng lặp qua từng Model (GPT-3.5, etc.)
    for model_name in MODELS_CONFIG:
        print(f"\n" + "█"*60)
        print(f"🤖 ĐANG CHẠY MODEL: {model_name}")
        print("█"*60)
        
        for ds_name, file_name in DATASETS_TO_RUN.items():
            print(f"\n👉 DATASET: {ds_name}")
            
            input_path = os.path.join(BASE_DIR, "datasets/ETT-small", file_name)
            output_dir = os.path.join(BASE_DIR, f"output/{ds_name}")
            
            # Tên file kết quả gắn liền với tên model để tránh ghi đè
            safe_model_name = model_name.replace("/", "-")
            output_file = os.path.join(output_dir, f"results_{ds_name}_{safe_model_name}.pkl")
            
            if not os.path.exists(input_path):
                print(f"❌ Không tìm thấy file: {input_path}")
                continue
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Check nếu đã chạy rồi thì bỏ qua (Tiết kiệm tiền API)
            if os.path.exists(output_file):
                print(f"   ⚠️ File kết quả đã tồn tại: {output_file}. Bỏ qua để tiết kiệm API.")
                # continue # Bỏ comment dòng này nếu muốn skip file đã chạy

            df, target_cols = load_and_clean_data(input_path)
            ds_results = {}
            
            for col in target_cols:
                print(f"   ... Đang dự báo cột: {col}")
                
                series = df[col]
                limit_size = 2000 
                test_size = 100
                
                if len(series) > limit_size: series = series.iloc[-limit_size:]
                train = series.iloc[:-test_size]
                test = series.iloc[-test_size:]
                
                try:
                    # Gọi API OpenAI
                    # Hàm này sẽ tự nhận diện tên model là GPT và dùng API thay vì load local
                    pred_dict = get_llmtime_predictions_data(
                        train, test, 
                        model=model_name,
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
                    
                    # Nghỉ 1 chút để tránh lỗi Rate Limit của OpenAI
                    time.sleep(1) 

                except Exception as e:
                    print(f"      ❌ Lỗi cột {col}: {e}")
                    import traceback
                    traceback.print_exc()

            # Lưu kết quả
            with open(output_file, 'wb') as f:
                pickle.dump(ds_results, f)
            print(f"💾 Đã lưu: {output_file}")

    print("\n🎉🎉🎉 HOÀN TẤT!")

if __name__ == "__main__":
    run_all_datasets_gpt()