import streamlit as st
import joblib
import pandas as pd

# Load model, scaler và danh sách cột
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Các cột đầu vào
base_cols = ['Nha_ve_sinh', 'Cau_truc', 'Dien_tich (m2)', 'Dien_tich_su_dung (m2)', 'Hem_duong (m)']
quans = [col.replace("Quan_", "") for col in columns if col.startswith("Quan_")]

st.title("🧠 Dự đoán giá nhà tại TP.HCM")

# Nhập liệu người dùng
user_input = {}
for col in base_cols:
    user_input[col] = st.number_input(f"{col}", value=0.0)

selected_quan = st.selectbox("Chọn quận", quans)
user_input["Quan"] = selected_quan

# Tạo DataFrame từ input
input_df = pd.DataFrame([user_input])

# One-hot encode cho cột Quận
quan_encoded = pd.get_dummies(input_df["Quan"], prefix="Quan", dtype=int)
input_base = input_df[base_cols]
input_processed = pd.concat([input_base, quan_encoded], axis=1)

# Căn chỉnh thứ tự cột đúng với mô hình
input_aligned = input_processed.reindex(columns=columns, fill_value=0)

# Chuẩn hóa và dự đoán
input_scaled = scaler.transform(input_aligned)
prediction = model.predict(input_scaled)[0]

# Kết quả
st.subheader("💰 Giá nhà dự đoán:")
st.success(f"{prediction:,.2f} tỷ đồng")
