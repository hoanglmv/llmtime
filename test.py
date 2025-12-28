import torch

def check_gpu_memory():
    if not torch.cuda.is_available():
        print("❌ Không tìm thấy GPU nào (CUDA chưa sẵn sàng).")
        return

    num_gpus = torch.cuda.device_count()
    print(f"✅ Tìm thấy {num_gpus} GPU khả dụng với PyTorch:\n")

    for i in range(num_gpus):
        # Lấy tên GPU
        gpu_name = torch.cuda.get_device_name(i)
        
        # Lấy thông tin bộ nhớ (trả về bytes)
        # free: dung lượng còn trống
        # total: tổng dung lượng
        free, total = torch.cuda.mem_get_info(i)
        
        # Đổi đơn vị sang GB
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        used_gb = total_gb - free_gb
        
        print(f"🔹 GPU {i}: {gpu_name}")
        print(f"   • Tổng cộng : {total_gb:.2f} GB")
        print(f"   • Đang dùng : {used_gb:.2f} GB")
        print(f"   • Còn trống : {free_gb:.2f} GB  <-- Quan trọng nhất")
        print("-" * 30)

if __name__ == "__main__":
    check_gpu_memory()