import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import re
import ast

# --- Đường dẫn file ---
PROCESSED_FILE = r"C:\Users\tlinh\Desktop\processed_data_for_ml_pandas.csv"
TFIDF_MODEL_PATH = r"C:\Users\tlinh\Desktop\tfidf_vectorizer.joblib"
MLB_MODEL_PATH = r"C:\Users\tlinh\Desktop\mlb.joblib"
PREDICTION_MODEL_PATH = r"C:\Users\tlinh\Desktop\prediction_model.joblib"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def train_model():
    try:
        # --- Đọc dữ liệu huấn luyện ---
        df = pd.read_csv(PROCESSED_FILE, encoding='utf-8')
        df['all_user_answers'] = df['all_user_answers'].apply(ast.literal_eval)
        df['processed_answers'] = df['all_user_answers'].apply(lambda x: [preprocess_text(ans) for ans in x])
        df['combined_answers'] = df['processed_answers'].apply(lambda x: ' '.join(x))
        df['suggested_locations'] = df['suggested_locations'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # --- Huấn luyện TF-IDF ---
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        X = tfidf_vectorizer.fit_transform(df['combined_answers'])

        # --- Chuẩn bị nhãn ---
        y = df['suggested_locations']
        mlb = MultiLabelBinarizer()
        y_transformed = mlb.fit_transform(y)

        # --- Huấn luyện mô hình Random Forest đa nhãn ---
        random_forest = RandomForestClassifier(n_estimators=300, random_state=42) # Thử với 200 cây
        multi_output_model = MultiOutputClassifier(random_forest)
        multi_output_model.fit(X, y_transformed)

        # --- Lưu mô hình đã huấn luyện ---
        joblib.dump(tfidf_vectorizer, TFIDF_MODEL_PATH)
        joblib.dump(mlb, MLB_MODEL_PATH)
        joblib.dump(multi_output_model, PREDICTION_MODEL_PATH)
        print("Mô hình Random Forest đã được huấn luyện và lưu thành công!")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file: {PROCESSED_FILE}")
        return None, None, None
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình huấn luyện: {e}")
        return None, None, None

def predict_locations(user_input_string):
    # --- Load mô hình đã huấn luyện ---
    try:
        tfidf_vectorizer = joblib.load(TFIDF_MODEL_PATH)
        mlb = joblib.load(MLB_MODEL_PATH)
        prediction_model = joblib.load(PREDICTION_MODEL_PATH)
    except FileNotFoundError:
        return "Lỗi: Không tìm thấy file mô hình đã lưu."
    except Exception as e:
        return f"Lỗi khi load mô hình: {e}"

    # --- Tiền xử lý dữ liệu đầu vào ---
    user_answers_list = user_input_string.split('#')
    processed_input = [preprocess_text(ans) for ans in user_answers_list]
    combined_input = ' '.join(processed_input)
    X_new = tfidf_vectorizer.transform([combined_input])

    # --- Dự đoán ---
    y_pred_transformed = prediction_model.predict(X_new)
    predicted_labels = mlb.inverse_transform(y_pred_transformed)

    return predicted_labels[0]

if __name__ == "__main__":
    # --- Huấn luyện mô hình (chỉ cần chạy một lần) ---
    train_model()

    # --- Ví dụ sử dụng hàm dự đoán ---
    user_input = "Mùa xuân (tháng 3 - 5)#Núi/Rừng#Du lịch khám phá thiên nhiên"
    predicted_locations = predict_locations(user_input)
    print("\nDữ liệu đầu vào từ người dùng:", user_input)
    print("Các địa điểm du lịch được gợi ý:", predicted_locations)

    user_input_2 = "Tháng 12#Biển#Nghỉ dưỡng"
    predicted_locations_2 = predict_locations(user_input_2)
    print("\nDữ liệu đầu vào từ người dùng:", user_input_2)
    print("Các địa điểm du lịch được gợi ý:", predicted_locations_2)