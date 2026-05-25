import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(
    page_title="Hệ thống Tiên lượng Hồi sức Tích cực Lão khoa",
    page_icon="🏥",
    layout="wide"
)

# --- GIAO DIỆN CHÍNH ---
st.title("🏥 Hệ thống Tiên lượng ICU Lão khoa nâng cao (XGBoost)")
st.markdown("""
Ứng dụng hỗ trợ ra quyết định lâm sàng dành cho bệnh nhân lớn tuổi trong ICU. 
*Đơn vị đo lường tuân thủ hệ SI theo tiêu chuẩn ISO 15189:2022.*
""")

# --- MÔ PHỎNG DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH (Để App chạy độc lập) ---
# Trong thực tế, anh sẽ load file model .json hoặc .pkl đã train sẵn.
@st.cache_resource
def load_or_train_model():
    # Tạo dữ liệu giả lập gồm 1000 bệnh nhân lão khoa ICU để train nhanh mô hình mẫu
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(65, 95, n_samples),
        'Clinical_Frailty_Scale': np.random.randint(1, 9, n_samples),
        'Protein_Z': np.random.uniform(0.5, 4.0, n_samples),
        'Lactate_Clearance_6h': np.random.uniform(-20, 60, n_samples),
        'NLR_Ratio': np.random.uniform(1.0, 25.0, n_samples),
        'Cystatin_C': np.random.uniform(0.5, 5.0, n_samples),
        'APACHE_II': np.random.randint(5, 40, n_samples)
    }
    df = pd.DataFrame(data)
    # Giả lập tiêu chí đầu ra (Tử vong hoặc trở nặng dựa trên các trọng số lâm sàng)
    logit = (df['Age']*0.02 + df['Clinical_Frailty_Scale']*0.3 - df['Protein_Z']*0.5 
             - df['Lactate_Clearance_6h']*0.04 + df['NLR_Ratio']*0.08 + df['Cystatin_C']*0.4 
             + df['APACHE_II']*0.1 - 5)
    prob = 1 / (1 + np.exp(-logit))
    df['Outcome'] = np.where(prob > 0.5, 1, 0)
    
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X, y)
    
    # Tạo SHAP Explainer
    explainer = shap.TreeExplainer(model)
    return model, explainer, X.columns.tolist()

model, explainer, feature_names = load_or_train_model()

# --- THANH SIDEBAR: NHẬP LIỆU LÂM SÀNG ---
st.sidebar.header("📋 Thông số Bệnh nhân")

with st.sidebar.form("patient_data_form"):
    patient_id = st.text_input("Mã định danh bệnh nhân (ID)", value="BN-2026-001")
    
    st.markdown("### 🧓 Đặc điểm Lão khoa & Nền tảng")
    age = st.slider("Tuổi (Năm)", 65, 100, 78)
    cfs = st.slider("Thang đo Frailty (Clinical Frailty Scale - CFS)", 1, 9, 5, 
                    help="1: Rất khỏe mạnh, 9: Tiên lượng tử vong rất gần")
    apache_ii = st.number_input("Điểm APACHE II", min_value=0, max_value=60, value=22)
    
    st.markdown("### 🔬 Chỉ số Xét nghiệm (Hệ SI / ISO 15189)")
    protein_z = st.number_input("Nồng độ Protein Z (mg/L)", min_value=0.0, max_value=10.0, value=1.8, step=0.1,
                                help="Dữ liệu nghiên cứu dấu ấn đông máu")
    lactate_clearance = st.number_input("Tốc độ thanh thải Lactate sau 6h (%)", min_value=-100.0, max_value=100.0, value=15.0, step=1.0,
                                        help="(Lactate_0h - Lactate_6h) / Lactate_0h * 100")
    nlr = st.number_input("Tỷ lệ Neutrophil / Lymphocyte (NLR)", min_value=0.0, max_value=100.0, value=8.5, step=0.1,
                          help="Chỉ số viêm hệ thống cấp tính")
    cystatin_c = st.number_input("Cystatin C huyết thanh (mg/L)", min_value=0.0, max_value=10.0, value=1.6, step=0.1,
                                 help="Đánh giá chức năng thận chính xác cho người già teo cơ")
    
    submit_button = st.form_submit_button("Tính toán tiên lượng")

# --- XỬ LÝ DỮ LIỆU & HÌNH THÀNH DỰ BÁO ---
# Chuyển đổi dữ liệu nhập vào thành DataFrame
input_data = pd.DataFrame([[age, cfs, protein_z, lactate_clearance, nlr, cystatin_c, apache_ii]], 
                           columns=feature_names)

# Dự đoán xác suất
risk_prob = model.predict_proba(input_data)[0][1]

# --- HIỂN THỊ KẾT QUẢ TIÊN LƯỢNG ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Kết quả phân tầng nguy cơ")
    st.metric(label="Xác suất biến cố nặng / Tử vong", value=f"{risk_prob*100:.1f} %")
    
    # Phân tầng mức độ nguy cơ bằng Alert color
    if risk_prob < 0.3:
        st.success("🔴 Nguy cơ: THẤP")
    elif risk_prob < 0.6:
        st.warning("🟡 Nguy cơ: TRUNG BÌNH")
    else:
        st.error("🚨 Nguy cơ: CAO - Cần can thiệp tích cực")
        
    st.markdown(f"**Thời gian đánh giá:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown(f"**Mã bệnh nhân:** {patient_id}")

with col2:
    st.subheader("🧠 Giải thích mô hình bằng AI (SHAP Explanations)")
    st.write("Biểu đồ dưới đây chỉ rõ mức độ đóng góp (tích cực hay tiêu cực) của từng chỉ số lâm sàng vào kết quả dự báo của thuật toán XGBoost.")
    
    # Tính toán giá trị SHAP cho ca bệnh hiện tại
    shap_values = explainer(input_data)
    
    # Vẽ biểu đồ SHAP Force Plot hoặc Waterfall Plot sử dụng matplotlib
    fig, ax = plt.subplots(figsize=(8, 4))
    # Đổi tên hiển thị sang tiếng Việt trên biểu đồ cho trực quan
    display_labels = [
        f"Tuổi ({age})", 
        f"Điểm CFS ({cfs})", 
        f"Protein Z ({protein_z} mg/L)", 
        f"Thanh thải Lactate ({lactate_clearance} %)", 
        f"Tỷ lệ NLR ({nlr})", 
        f"Cystatin C ({cystatin_c} mg/L)", 
        f"APACHE II ({apache_ii})"
    ]
    
    shap.plots.bar(shap_values[0], max_display=7, show=False)
    plt.title("Trọng số ảnh hưởng của các chỉ số lâm sàng", fontsize=12, pad=20)
    plt.tight_layout()
    st.pyplot(fig)

# --- TẦNG QUẢN LÝ DỮ LIỆU & LƯU CHUỖI THỜI GIAN ---
st.markdown("---")
st.subheader("💾 Nhật ký theo dõi chuỗi thời gian (Time-series Logging)")
st.write("Dữ liệu này có thể cấu hình để tự động đẩy ngược lại hệ thống thu thập dữ liệu chuỗi thời gian hoặc lưu vào bệnh án điện tử.")

# Tạo bảng dữ liệu lịch sử mô phỏng để bác sĩ xem cấu trúc chuỗi thời gian
if 'history' not in st.session_state:
    st.session_state.history = []

if submit_button:
    st.session_state.history.append({
        "Thời gian": datetime.now().strftime("%H:%M:%S"),
        "ID": patient_id,
        "Tuổi": age,
        "CFS": cfs,
        "Protein Z (mg/L)": protein_z,
        "Lactate Clearance (%)": lactate_clearance,
        "NLR": nlr,
        "Cystatin C (mg/L)": cystatin_c,
        "APACHE II": apache_ii,
        "Xác suất nguy cơ": f"{risk_prob*100:.1f}%"
    })

if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history))
else:
    st.info("Nhấn nút 'Tính toán tiên lượng' ở thanh bên để ghi nhận điểm dữ liệu vào chuỗi thời gian.")