from tf_keras.applications.mobilenet_v2 import preprocess_input
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tf_keras as keras
import google.generativeai as genai

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(
    page_title="PC Hardware AI Assistant",
    page_icon="💻",
    layout="wide"
)

# --- NẠP CẤU HÌNH & MÔ HÌNH (CACHED) ---
@st.cache_resource
def load_resources():
    # 1. Nạp API Key
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    api_key = config.get("GEMINI_API_KEY", "AQ.Ab8RN6KDdBuVWoeA99pk0AZiayovvQYZM9Jk9Vyjsqp0GY9o7Q").strip()
    if not api_key:
        st.error("❌ Chưa nhập GEMINI_API_KEY trong file config.json!")
        st.stop()

    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)

    # 2. Khởi tạo mô hình Gemini
    gemini_model = genai.GenerativeModel('gemini-3.5-flash-lite')

    

    # 3. Nạp mô hình CNN
    cnn_model = keras.models.load_model('models/hardware_cnn_model.h5')
    
    # 4. Nạp CSV dữ liệu linh kiện
    df_hardware = pd.read_csv('linh-kien-pc.csv')
    
    return gemini_model, cnn_model, df_hardware

# KẾT NỐI BIẾN TOÀN CỤC
try:
    gemini_model, cnn_model, df_hardware = load_resources()
    CLASS_NAMES = ['CPU', 'GPU', 'Mainboard', 'PSU', 'RAM', 'SSD']
except Exception as e:
    st.error(f"❌ Lỗi khởi tạo ứng dụng: {e}")
    st.stop()

# --- HÀM DỰ ĐOÁN ẢNH BẰNG CNN ---
def predict_image(image):
    img_resized = image.convert('RGB').resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    
    # BẮT BUỘC: Thay vì chia cho 255.0, hãy dùng hàm preprocess_input của MobileNetV2
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = cnn_model.predict(img_array, verbose=0)
    pred_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][pred_idx])
    return CLASS_NAMES[pred_idx], confidence

# --- HÀM TƯ VẤN GEMINI (ĐÃ TỐI ƯU TRÁNH LẶP LẠI NHẬN DIỆN ÁNH) ---
def ask_gemini(prompt, category=None, confidence=None, image=None, is_first_image_query=False):
    context = ""
    if category:
        conf_str = f" với độ chính xác {confidence*100:.1f}%" if confidence else ""
        context += f"\n[Mô hình CNN nhận diện ban đầu là: {category}{conf_str}]\n"
        
        cat_col = None
        for col in df_hardware.columns:
            if col.strip().lower() in ['category', 'loai', 'loailinhkien', 'type', 'nhom']:
                cat_col = col
                break
        
        if cat_col:
            context += f"CSDL linh kiện cửa hàng hiện có:\n{df_hardware.to_string(index=False)}\n"

    system_prompt = f"""
    Bạn là một chuyên gia tư vấn phần cứng PC thông minh.

    QUY TẮC XỬ LÝ QUAN TRỌNG:
    1. KIỂM TRA LẠI NHẬN DIỆN:
    - Mô hình CNN dự đoán nhóm linh kiện là: '{category}' ({f'{confidence*100:.1f}%' if confidence else ''}).
    - NẾU BẠN NHÌN ẢNH VÀ THẤY CNN ĐOÁN SAI, HÃY TỰ ĐÍNH CHÍNH LẠI: "Dựa vào hình ảnh, đây là [Tên linh kiện đúng] chứ không phải {category}...".
    - Đọc chính xác thương hiệu, tên dòng sản phẩm in trên ảnh.

    2. TƯ VẤN THÔNG TIN:
    - Tra cứu giá và thông số từ CSDL hoặc báo giá trung bình thị trường.
    - Trả lời ngắn gọn, lịch sự bằng tiếng Việt.

    Ngữ cảnh CSDL:
    {context}
    """
    
    full_prompt = f"{system_prompt}\n\nCâu hỏi người dùng: {prompt}"
    
    try:
        if image is not None and is_first_image_query:
            response = gemini_model.generate_content([image, full_prompt])
        else:
            response = gemini_model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        err_str = str(e)
        if "429" in err_str:
            return "⚠️ **Hệ thống AI đang bận hoặc hết hạn ngạch.** Vui lòng thử lại sau giây lát!"
        return f"⚠️ **Lỗi kết nối AI:** {e}"

# --- GIAO DIỆN CHÍNH ---
st.title("💻 PC Hardware AI Assistant")
st.caption("Trợ lý AI phân loại linh kiện máy tính và tư vấn cấu hình PC")

# Khởi tạo session state lưu lịch sử chat và ảnh
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "detected_category" not in st.session_state:
    st.session_state.detected_category = None
if "image_analyzed" not in st.session_state:
    st.session_state.image_analyzed = False

    if "detected_confidence" not in st.session_state:
        st.session_state.detected_confidence = None

# Thanh Sidebar tải ảnh
with st.sidebar:
    st.header("📸 Tải ảnh linh kiện")
    uploaded_file = st.file_uploader("Chọn ảnh linh kiện PC...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Nếu người dùng chọn ảnh mới
        if st.session_state.uploaded_file_name != uploaded_file.name:
            image = Image.open(uploaded_file)
            st.session_state.current_image = image
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.image_analyzed = False  # Đặt lại cờ: ảnh mới chưa phân tích
            
            with st.spinner("CNN đang nhận diện nhóm linh kiện..."):
                category, conf = predict_image(image)
                st.session_state.detected_category = category
        
        st.image(st.session_state.current_image, caption="Ảnh đã tải lên", use_container_width=True)
        st.success(f"**Nhóm linh kiện:** {st.session_state.detected_category}")

# Hiển thị tin nhắn chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ô nhập dữ liệu từ người dùng
if user_prompt := st.chat_input("Hỏi AI về linh kiện hoặc cấu hình PC..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI đang suy luận câu trả lời..."):
            # Kiểm tra xem đây có phải là lượt đầu tiên hỏi về ảnh mới không
            is_first_time = False
            if st.session_state.current_image is not None and not st.session_state.image_analyzed:
                is_first_time = True
                st.session_state.image_analyzed = True # Đã phân tích xong lượt đầu
            
            reply = ask_gemini(
                user_prompt, 
                st.session_state.detected_category, 
                st.session_state.detected_confidence,
                st.session_state.current_image,
                is_first_image_query=is_first_time
            )
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})