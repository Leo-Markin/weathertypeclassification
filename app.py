import pandas as pd
import streamlit as st

from model import predict, load_model, open_data


def show_main_page():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Прогноз погоды",
        page_icon="🌦️",
    )

    st.write(
        """
        # 🌦️ Классификация типа погоды
        **Дождливо (Rainy)**, **Солнечно (Sunny)**, **Облачно (Cloudy)** или
        **Снежно (Snowy)**.

        Задайте параметры на боковой панели слева и получите прогноз.
        """
    )


def sidebar_input_features():
    st.sidebar.header("Параметры погоды")
    season = st.sidebar.selectbox(
        "Сезон", ("Winter", "Spring", "Summer", "Autumn"))
    cloud_cover = st.sidebar.selectbox(
        "Облачность", ("clear", "partly cloudy", "cloudy", "overcast"))
    location = st.sidebar.radio(
        "Тип местности", ("inland", "mountain", "coastal"))
    temperature = st.sidebar.slider(
        "Температура, °C", min_value=-30.0, max_value=60.0, value=20.0, step=0.5)
    humidity = st.sidebar.slider(
        "Влажность, %", min_value=0, max_value=100, value=70, step=1)
    wind_speed = st.sidebar.slider(
        "Скорость ветра, км/ч", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
    precipitation = st.sidebar.slider(
        "Осадки, %", min_value=0, max_value=100, value=50, step=1)
    pressure = st.sidebar.number_input(
        "Атмосферное давление, гПа", min_value=870.0, max_value=1085.0,
        value=1013.0, step=0.5)
    uv_index = st.sidebar.slider(
        "УФ-индекс", min_value=0, max_value=14, value=4, step=1)
    visibility = st.sidebar.slider(
        "Видимость, км", min_value=0.0, max_value=20.0, value=5.0, step=0.5)

    data = {
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind Speed": wind_speed,
        "Precipitation (%)": precipitation,
        "Cloud Cover": cloud_cover,
        "Atmospheric Pressure": pressure,
        "UV Index": uv_index,
        "Season": season,
        "Visibility (km)": visibility,
        "Location": location,
    }
    return pd.DataFrame(data, index=[0])


def write_user_data(df):
    st.write("## Заданные параметры")
    st.write(df)


def write_prediction(prediction, proba_df):
    emoji = {"Rainy": "🌧️", "Sunny": "☀️", "Cloudy": "☁️", "Snowy": "❄️"}
    st.write("## Прогноз")
    st.success(f"{emoji.get(prediction, '')} Тип погоды: **{prediction}**")

    st.write("## Вероятности по классам")
    st.bar_chart(proba_df.set_index("Тип погоды")["Вероятность"])
    st.write(proba_df)


def process_main_page():
    show_main_page()

    # Информация о качестве модели
    try:
        bundle = load_model()
    except FileNotFoundError:
        st.error(
            "Файл модели не найден.")
        return

    st.info(f"Точность модели (accuracy на тесте): **{bundle['accuracy']:.3f}**")

    user_df = sidebar_input_features()
    write_user_data(user_df)

    prediction, proba_df = predict(user_df)
    write_prediction(prediction, proba_df)


if __name__ == "__main__":
    process_main_page()
