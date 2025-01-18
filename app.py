# Импорт необходимых библиотек
import streamlit as st
import requests
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas

# Заголовок приложения
st.title("Классификация изображений")

# Опция для выбора режима ввода изображения
mode = st.radio("Выберите способ ввода изображения:", ("Загрузить изображение", "Нарисовать изображение"))

if mode == "Загрузить изображение":
    # Загрузка изображения
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Отображение загруженного изображения
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_column_width=True)

        # Автоматическая классификация без кнопки
        # Подготовка изображения для отправки
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Отправка запроса на сервер FastAPI
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict/",
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

elif mode == "Нарисовать изображение":
    # Настройки для canvas
    stroke_width = st.slider("Толщина линии:", 1, 25, 9)
    stroke_color = st.color_picker("Цвет линии:", "#FFFFFF")
    bg_color = st.color_picker("Цвет фона:", "#000000")
    realtime_update = st.checkbox("Обновлять в реальном времени", True)

    # Создание колонок для размещения изображения и прогноза рядом
    col1, col2 = st.columns(2)

    with col1:
        # Создание canvas для рисования
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            update_streamlit=realtime_update,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

    if canvas_result.image_data is not None:
        # Преобразование данных холста в изображение
        image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')

        # Подготовка изображения для отправки
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Отправка запроса на сервер FastAPI
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict/",
                files={"file": ("image.png", img_byte_arr, "image/png")}
            )
            response.raise_for_status()

            # Получение и отображение результата в правой колонке
            prediction = response.json()["prediction"]
            with col2:
                st.write(f"**Прогноз классификации:** {prediction}")

        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка при отправке запроса: {e}")
        except ValueError as e:
            st.error(f"Ошибка при обработке ответа: {e}")