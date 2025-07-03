import streamlit as st, pandas as pd, joblib, os

# ==== Load mô hình / scaler / cột ==========================================================
model   = joblib.load("model.pkl")
scaler  = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ==== Xác định cột cơ bản & cột quận =======================================================
base_cols = ['Nha_ve_sinh','Cau_truc','Dien_tich (m2)',
             'Dien_tich_su_dung (m2)','Hem_duong (m)']
quan_cols = [c for c in columns if c not in base_cols]

pretty_names  = [c.replace("_"," ") for c in quan_cols]
pretty2raw    = dict(zip(pretty_names, quan_cols))

# ==== Giao diện nhập ======================================================================
st.title("📊 Dự đoán giá nhà TP.HCM")

nha  = st.number_input("🛁 Số nhà vệ sinh",          0.0, step=1.0,
                       help="Tổng số phòng vệ sinh trong nhà.")
cau  = st.number_input("🏠 Số tầng (cấu trúc)",       0.0, step=1.0,
                       help="Số tầng thực của ngôi nhà.")
dt   = st.number_input("📏 Diện tích đất (m²)",       0.0, step=1.0,
                       help="Diện tích lô đất (mét vuông).")
dtsd = st.number_input("📐 Diện tích sử dụng (m²)",   0.0, step=1.0,
                       help="Tổng diện tích sàn sử dụng (mét vuông).")
hem  = st.number_input("🛣️ Hẻm/Đường trước nhà (m)", 0.0, step=0.5,
                       help="Độ rộng hẻm hoặc đường trước nhà (mét).")

chon_quan = st.multiselect("🏙️ Chọn quận/huyện (có thể chọn nhiều):",
                           pretty_names,
                           help="Có thể chọn 1 hoặc nhiều quận để dự đoán / so sánh.")

# ==== Hàm dựng DataFrame đúng cột ==========================================================
def make_X(quans):
    """trả về DataFrame 1 hàng với quận/quận­huyện được gán one-hot"""
    row = {
        'Nha_ve_sinh': nha,
        'Cau_truc': cau,
        'Dien_tich (m2)': dt,
        'Dien_tich_su_dung (m2)': dtsd,
        'Hem_duong (m)': hem,
        **{q:0 for q in quan_cols}
    }
    for q in quans:
        row[pretty2raw[q]] = 1
    return (pd.DataFrame([row])
            .reindex(columns=columns, fill_value=0))

# ==== 1) Dự đoán giá trung bình nếu chọn nhiều quận ========================================
if st.button("📈 Dự đoán giá"):
    if not chon_quan:
        st.warning("⚠️ Hãy chọn ít nhất một quận/huyện!")
    else:
        X = make_X(chon_quan)
        price = model.predict(scaler.transform(X))[0]
        st.success(f"💰 Giá nhà dự đoán: **{price:,.2f} tỷ đồng** "
                   f"(trung bình trên {len(chon_quan)} quận)")

# ==== 2) So sánh giá từng quận riêng lẻ ====================================================
if st.button("📊 So sánh giá các quận đã chọn"):
    if not chon_quan:
        st.warning("⚠️ Hãy chọn ít nhất một quận/huyện để so sánh!")
    else:
        results = []
        for q in chon_quan:
            Xq = make_X([q])
            price_q = model.predict(scaler.transform(Xq))[0]
            results.append({"Quận/Huyện": q, "Giá dự đoán (tỷ)": round(price_q, 2)})
        df_res = pd.DataFrame(results)

        st.subheader("🔎 Bảng so sánh")
        st.dataframe(df_res, use_container_width=True)

        st.subheader("📊 Biểu đồ")
        st.bar_chart(df_res.set_index("Quận/Huyện"))