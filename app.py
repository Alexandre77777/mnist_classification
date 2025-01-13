import streamlit as st
import requests
from PIL import Image
import io


# Заголовок приложения
st.title("Классификация изображений")

# Загрузка изображения
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Отображение загруженного изображения
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    # Кнопка для запуска классификации
    if st.button("Классифицировать"):
        # Подготовка изображения для отправки
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Отправка запроса на сервер FastAPI
        try:
            response = requests.post(
                "https://classification-mf1z.onrender.com/predict/",
                files={"file": ("image.png", img_byte_arr, "image/png")}
            )
            response.raise_for_status()
            
            # Получение и отображение результата
            prediction = response.json()["prediction"]
            st.write(f"Прогноз классификации: {prediction}")
        
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка при отправке запроса: {e}")
        except ValueError as e:
            st.error(f"Ошибка при обработке ответа: {e}")