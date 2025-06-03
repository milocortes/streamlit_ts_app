# Core Pkgs
import streamlit as st 
st.set_page_config(page_title="PIB Geoespacial", page_icon="🌻", layout="centered", initial_sidebar_state="auto")

# Data Packages
import pandas as pd 

# Visualization Packages
import plotly.express as px

def main():
    """Geospatial GDP web app with Streamlit"""

    st.title("PIB Geoespacial")

    activity = ["Home Page", "Pronóstico", "Visualizador de Datos", "Data Overview", "About"]

    choice = st.sidebar.selectbox("Menu", activity)

    if choice == "Home Page" : 
        st.subheader("Estimación del GDP de El Salvador usando covariables ambientales")
        st.write()
    
    if choice == "Pronóstico":
        st.subheader("Modelos de Pronóstico")
        st.write("")

        tab1_forecast, tab2_forecast = st.tabs(["Tasa de Crecimiento del PIB", "PIB en Niveles"])

        with tab1_forecast:
            tc_forecast = st.selectbox("Modelo de Aprendizaje de Máquina", ["XGBoost", "Random Forest", "Elastic Net", "Ensamble"])
        with tab2_forecast:
            level_forecast = st.selectbox("Modelo de Aprendizaje de Máquina", ["ARIMA"])
    

    if choice == "Visualizador de Datos":
        st.subheader("Visualizador de Datos")
        st.write("")


        df = px.data.gapminder()

        fig = px.scatter(
            df.query("year==2007"),
            x="gdpPercap",
            y="lifeExp",
            size="pop",
            color="continent",
            hover_name="country",
            log_x=True,
            size_max=60,
        )

        ts_datos = pd.read_excel("data/dataset_clean_slv.xlsx", sheet_name="DATOS").iloc[-41:]

        occupation = st.multiselect("Serie", ['pib_bc_usd', 'viirs_bm_mean', 'co2'])

        fig = px.line(ts_datos, x="anio_trim", y=occupation)
        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
        with tab1:
            # Use the Streamlit theme.
            # This is the default. So you can also omit the theme argument.
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        with tab2:
            # Use the native Plotly theme.
            st.plotly_chart(fig, theme=None, use_container_width=True)
            
    if choice == "Data Overview":
        st.subheader("Data Overview")
        st.write("Los modelos incluyen el siguiente listado de covariables:")
        
        recursos = pd.read_excel("data/dataset_clean_slv.xlsx")[["descripcion",  "fuente", "enlace"]]

        recursos.columns = ["Variable", "Fuente", "Enlace"]

        st.table(recursos)

    if choice == "About":
        st.subheader("About")
        st.write("")

        st.markdown("""
        ### Geospatial GDP web app with Streamlit
        
        for info:
        - [streamlit](https://streamlit.io)
        """)


if __name__ == "__main__":
    main()
