import os
import torch
import matplotlib
# Chuyển backend matplotlib sang 'Agg' để không yêu cầu màn hình hiển thị (tránh lỗi trên server)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openai

# --- CẤU HÌNH MÔI TRƯỜNG ---
os.environ['OMP_NUM_THREADS'] = '4'

# Đã comment lại dòng này để không bắt buộc phải có API Key
# openai.api_key = os.environ['OPENAI_API_KEY']
# openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

from data.serialize import SerializerSettings
from models.utils import grid_iter
from models.promptcast import get_promptcast_predictions_data
from models.darts import get_arima_predictions_data
from models.llmtime import get_llmtime_predictions_data
from data.small_context import get_datasets
from models.validation_likelihood_tuning import get_autotuned_predictions_data

def plot_preds(train, test, pred_dict, model_name, show_samples=False):
    pred = pred_dict['median']
    pred = pd.Series(pred, index=test.index)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train)
    plt.plot(test, label='Truth', color='black')
    plt.plot(pred, label=model_name, color='purple')
    # shade 90% confidence interval
    samples = pred_dict['samples']
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    plt.fill_between(pred.index, lower, upper, alpha=0.3, color='purple')
    if show_samples:
        samples = pred_dict['samples']
        # convert df to numpy array
        samples = samples.values if isinstance(samples, pd.DataFrame) else samples
        for i in range(min(10, samples.shape[0])):
            plt.plot(pred.index, samples[i], color='purple', alpha=0.3, linewidth=1)
    plt.legend(loc='upper left')
    if 'NLL/D' in pred_dict:
        nll = pred_dict['NLL/D']
        if nll is not None:
            plt.text(0.03, 0.85, f'NLL/D: {nll:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
    # --- THAY ĐỔI QUAN TRỌNG: Lưu file ảnh thay vì show ---
    filename = f"prediction_{model_name}.png"
    plt.savefig(filename)
    print(f"Đã lưu biểu đồ vào file: {filename}")
    plt.close() # Đóng figure để giải phóng bộ nhớ

# In thông tin bộ nhớ GPU (nếu có)
if torch.cuda.is_available():
    print(f"CUDA Available. Memory: {torch.cuda.max_memory_allocated()}")
else:
    print("Running on CPU (No CUDA detected)")

# --- CÁC CẤU HÌNH MODEL ---

# Cấu hình cho GPT-2 (Model nhỏ, miễn phí, chạy offline)
gpt2_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False, 
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)

# Các cấu hình cũ (giữ lại để tham khảo)
gpt4_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1.0,
    top_p=0.8,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-')
)

gpt3_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)

# --- DANH SÁCH MODEL ---
model_hypers = {
     'GPT-2': {'model': 'gpt2', **gpt2_hypers}, # Model chính chúng ta sẽ dùng
     # 'LLMTime GPT-3.5': {'model': 'gpt-3.5-turbo-instruct', **gpt3_hypers},
     # 'LLMTime GPT-4': {'model': 'gpt-4', **gpt4_hypers},
 }

# --- CHỌN MODEL ĐỂ CHẠY ---
model_predict_fns = {
    'GPT-2': get_llmtime_predictions_data
}

model_names = list(model_predict_fns.keys())

# --- LOAD DỮ LIỆU ---
datasets = get_datasets()
ds_name = 'AirPassengersDataset' # Dữ liệu hành khách máy bay (kinh điển)

if ds_name in datasets:
    data = datasets[ds_name]
    train, test = data 
    out = {}

    print(f"Bắt đầu chạy dự báo với model: {model_names} trên dữ liệu {ds_name}...")

    for model in model_names: 
        print(f"Dang chay model {model}...")
        model_hypers[model].update({'dataset_name': ds_name}) 
        hypers = list(grid_iter(model_hypers[model]))
        num_samples = 10
        
        # Hàm này sẽ tự động chạy model và tính toán
        pred_dict = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=True, parallel=False)
        
        out[model] = pred_dict
        # Vẽ và lưu biểu đồ
        plot_preds(train, test, pred_dict, model, show_samples=True)
        
    print("HOÀN TẤT! Hãy kiểm tra file ảnh .png vừa được tạo ra trong thư mục.")
else:
    print(f"Lỗi: Không tìm thấy dataset {ds_name}")