import os
import pandas as pd
from datetime import datetime

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Đường dẫn đến thư mục chứa file .tsf
DATASET_DIR = "/home/myvh07/hoanglmv/Project/llmtime/datasets" 

def convert_tsf_to_dataframe(full_file_path_and_name, replace_missing_vals_with="NaN", value_column_name="series_value"):
    """
    Hàm đọc file .tsf (được cung cấp bởi Monash Repository)
    """
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, 'r', encoding='cp1252') as file:
        for line in file:
            line = line.strip()
            if line:
                if line.startswith("@attribute"):
                    col_names.append(line.split()[1])
                    col_types.append(line.split()[2])
                if line.startswith("@frequency"):
                    frequency = line.split()[1]
                if line.startswith("@horizon"):
                    forecast_horizon = int(line.split()[1])
                if line.startswith("@missing"):
                    contain_missing_values = bool(line.split()[1])
                if line.startswith("@equallength"):
                    contain_equal_length = bool(line.split()[1])

            if not found_data_tag:
                if line.startswith("@data"):
                    found_data_tag = True
            else:
                if line and not started_reading_data_section:
                    started_reading_data_section = True
                    found_data_section = True
                    all_series = []
                    for col in col_names:
                        all_data[col] = []

                if found_data_section:
                    # Xử lý dữ liệu từng dòng
                    parts = line.split(":")
                    if len(parts) >= 2: # Đảm bảo có phần data
                        # Phần metadata (tên series, start_timestamp...)
                        meta_part = parts[0].split(",")
                        # Phần giá trị chuỗi thời gian
                        series_part = parts[-1].split(",")
                        
                        # Mapping metadata vào cột tương ứng
                        # Lưu ý: Cấu trúc tsf có thể khác nhau số lượng attribute
                        # Đoạn này xử lý linh hoạt cho các attribute cơ bản
                        for idx, val in enumerate(meta_part):
                             if idx < len(col_names) - 1: # Trừ cột series_value cuối cùng
                                 all_data[col_names[idx]].append(val)
                        
                        # Xử lý missing values
                        clean_series = []
                        for val in series_part:
                            if val == "?":
                                clean_series.append(replace_missing_vals_with)
                            else:
                                clean_series.append(val)
                        
                        # Lưu chuỗi số vào cột cuối cùng
                        # Lưu dưới dạng string "val1,val2,..." để CSV không bị vỡ dòng
                        all_data[col_names[-1]].append(",".join(clean_series))
                        
                    line_count += 1

    if line_count == 0:
        print("⚠️ Không tìm thấy dữ liệu trong file.")
        return None

    # Tạo DataFrame
    df = pd.DataFrame(all_data)
    
    # Thêm thông tin Frequency vào tên file hoặc metadata nếu cần
    print(f"   ℹ️ Frequency: {frequency}, Horizon: {forecast_horizon}")
    
    return df

def process_conversion():
    # Kiểm tra đường dẫn
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Không tìm thấy thư mục: {DATASET_DIR}")
        return

    print(f"📂 Đang quét thư mục: {DATASET_DIR}")
    
    count = 0
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith(".tsf"):
                tsf_path = os.path.join(root, file)
                csv_filename = file.replace(".tsf", ".csv")
                csv_path = os.path.join(root, csv_filename)
                
                print(f"\n🔄 Đang xử lý: {file}...")
                try:
                    df = convert_tsf_to_dataframe(tsf_path)
                    
                    if df is not None:
                        # Lưu ra CSV
                        df.to_csv(csv_path, index=False)
                        print(f"   ✅ Đã tạo: {csv_filename}")
                        count += 1
                except Exception as e:
                    print(f"   ❌ Lỗi khi chuyển đổi file {file}: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"\n🎉 HOÀN TẤT! Đã chuyển đổi thành công {count} file.")

if __name__ == "__main__":
    process_conversion()