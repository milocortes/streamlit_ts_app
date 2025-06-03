import streamlit as st

from lib.utils.load import load_config

# Page config
#st.set_page_config(page_title="Pronóstico PIB - El Salvador", layout="wide")

# Load config
config, instructions, readme = load_config(
    "config_streamlit.toml", "config_instructions.toml", "config_readme.toml"
)


st.sidebar.title("1. Data")

# Load data
with st.sidebar.expander("Dataset", expanded=True):

    file = st.file_uploader("Upload a csv file", type="csv", help=readme["tooltips"]["dataset_upload"])

st.sidebar.title("2. Modelling")

st.sidebar.title("3. Evaluation")

st.sidebar.title("4. Forecast")