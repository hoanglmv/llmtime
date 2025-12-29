import os

import pandas as pd

import numpy as np

import pickle

import matplotlib.pyplot as plt

from dotenv import load_dotenv



# --- CẤU HÌNH HỆ THỐNG ---

load_dotenv() 



# Ép dùng GPU 1

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['OMP_NUM_THREADS'] = '4'



try:

    from huggingface_hub import login

    hf_token = os.getenv("HF_TOKEN")

    if hf_token: login(token=hf_token)

except: pass



import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from data.serialize import SerializerSettings

from models.llmtime import get_llmtime_predictions_data



BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Đảm bảo đọc file đã làm sạch (cleaned)

DATA_PATH = os.path.join(BASE_DIR, "datasets/ETT-small/ETTm1.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "output/ETTm1")



os.makedirs(OUTPUT_DIR, exist_ok=True)



# Cấu hình Model (Dictionary chứa settings)

llama_hypers = dict(

    temp=0.7,

    alpha=0.95,

    beta=0.3,

    basic=False,

    settings=SerializerSettings(base=10, prec=2, signed=True, half_bin_correction=True)

)



def run_ettm1_inference():

    print(f"🚀 Đang đọc dữ liệu từ: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):

        print(f"❌ Lỗi: Không tìm thấy file {DATA_PATH}. Hãy chạy run/preprocess_ETTm1.py trước!")

        return



    # Đọc dữ liệu (Đã sạch)

    df = pd.read_csv(DATA_PATH)

    

    # Ép kiểu lại date để đảm bảo không lỗi

    if 'date' in df.columns:

        df['date'] = pd.to_datetime(df['date'])

    

    target_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

    all_results = {}



    for col in target_cols:

        print(f"\n" + "="*50)

        print(f"🔄 ĐANG DỰ BÁO CỘT: {col}")

        print("="*50)



        if col not in df.columns:

            print(f"⚠️ Cảnh báo: Không tìm thấy cột {col} trong dữ liệu.")

            continue



        series = df[col]

        

        # Lấy 2000 dòng cuối

        limit_size = 2000 

        test_size = 100

        

        if len(series) > limit_size:

            series = series.iloc[-limit_size:]

        

        train = series.iloc[:-test_size]

        test = series.iloc[-test_size:]

        

        print(f"   - Train size: {len(train)}")

        print(f"   - Test size:  {len(test)}")



        try:

            # --- [ĐOẠN SỬA LỖI QUAN TRỌNG] ---

            # Thay vì truyền 'model_hypers=...', ta truyền thẳng tham số vào.

            # **llama_hypers sẽ tự động giải nén 'settings', 'temp', 'alpha'... vào hàm.

            

            pred_dict = get_llmtime_predictions_data(

                train, test, 

                model='llama-7b',   # <--- Truyền tên model trực tiếp

                num_samples=10,

                **llama_hypers      # <--- Unpack settings và các tham số khác từ dict

            )

            

            all_results[col] = {

                'train': train,

                'test': test,

                'pred_median': pred_dict['median'],

                'pred_samples': pred_dict['samples']

            }

            print(f"✅ Hoàn thành dự báo cột {col}")



        except Exception as e:

            print(f"❌ Lỗi khi chạy cột {col}: {e}")

            import traceback

            traceback.print_exc()



    output_file = os.path.join(OUTPUT_DIR, "results_ETTm1.pkl")

    with open(output_file, 'wb') as f:

        pickle.dump(all_results, f)

    

    print(f"\n🎉 ĐÃ XONG! Kết quả lưu tại: {output_file}")

    print("👉 Hãy chạy file util/check_ETTm1.py để xem biểu đồ.")



if __name__ == "__main__":

    run_ettm1_inference()