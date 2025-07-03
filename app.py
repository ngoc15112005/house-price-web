import streamlit as st, pandas as pd, joblib, os

# ---- load ---
model   = joblib.load("model.pkl")      # LinearRegression
scaler  = joblib.load("scaler.pkl")     # RobustScaler đã fit
columns = joblib.load("columns.pkl")    # list cột đúng

# ---- danh sách quận (từ columns.pkl) ---
quan_cols = [c for c in columns if c not in
             ['Nha_ve_sinh','Cau_truc','Dien_tich (m2)',
              'Dien_tich_su_dung (m2)','Hem_duong (m)']]

pretty = [c.replace("_"," ") for c in quan_cols]
map_pretty = dict(zip(pretty, quan_cols))

# ---- UI ----
st.title("📊 Dự đoán giá nhà TP.HCM")

nha  = st.number_input("Nha_ve_sinh", 0.0, step=1.0, value=3.0)
cau  = st.number_input("Cau_truc",      0.0, step=1.0, value=3.0)
dt   = st.number_input("Dien_tich (m2)", 0.0, step=1.0, value=43.0)
dtsd = st.number_input("Dien_tich_su_dung (m2)", 0.0, step=1.0, value=115.0)
hem  = st.number_input("Hem_duong (m)", 0.0, step=0.5, value=4.0)
quan = st.selectbox("Chọn quận", pretty, index=pretty.index("Quận 8"))

if st.button("Dự đoán"):
    # base
    data = {'Nha_ve_sinh':nha,'Cau_truc':cau,'Dien_tich (m2)':dt,
            'Dien_tich_su_dung (m2)':dtsd,'Hem_duong (m)':hem}
    # one-hot
    for q in quan_cols:
        data[q] = 1 if q == map_pretty[quan] else 0

    X = pd.DataFrame([data]).reindex(columns=columns, fill_value=0)
    X_scaled = scaler.transform(X)           # dùng scaler đã fit
    price = model.predict(X_scaled)[0]

    st.success(f"💰 Giá nhà dự đoán: **{price:,.2f} tỷ đồng**")