import pandas as pd
import re
import ast

# --- Đường dẫn file ---
INPUT_FILE = r"C:\Users\tlinh\Desktop\all_user_answers.txt"
OUTPUT_FILE = r"C:\Users\tlinh\Desktop\processed_data_for_ml_pandas.csv"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def create_processed_csv(input_file, output_file):
    try:
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('|')
                    if len(parts) == 2:
                        user_answers_str = parts[0].split('#')
                        suggested_locations_str = [loc.strip() for loc in parts[1].split(',')]

                        processed_answers = [preprocess_text(ans) for ans in user_answers_str]
                        combined_answers = ' '.join(processed_answers)

                        data.append({
                            'all_user_answers': user_answers_str,
                            'processed_answers': processed_answers,
                            'combined_answers': combined_answers,
                            'suggested_locations': suggested_locations_str
                        })
                    elif len(parts) == 1:
                        user_answers_str = parts[0].split('#')
                        processed_answers = [preprocess_text(ans) for ans in user_answers_str]
                        combined_answers = ' '.join(processed_answers)
                        data.append({
                            'all_user_answers': user_answers_str,
                            'processed_answers': processed_answers,
                            'combined_answers': combined_answers,
                            'suggested_locations': [] # Nếu không có địa điểm gợi ý
                        })
                    else:
                        print(f"Dòng không đúng định dạng: {line}")

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Đã tạo thành công file: {output_file}")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file: {input_file}")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    create_processed_csv(INPUT_FILE, OUTPUT_FILE)