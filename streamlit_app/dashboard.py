import streamlit as st

from lib.utils.load import load_config
from lib.inputs.eval import input_metrics, input_scope_eval
import pandas as pd 
import datetime

# Page config
#st.set_page_config(page_title="Pronóstico PIB - El Salvador", layout="wide")

# Load config
config, instructions, readme = load_config(
    "config_streamlit.toml", "config_instructions.toml", "config_readme.toml"
)

def main():
    # Info
    with st.expander(
        "App to build a time series forecasting model in a few clicks", expanded=False
    ):
        st.write(readme["app"]["app_intro"])
        st.write("")
    st.write("")

    st.sidebar.title("1. Data")

    # Load data
    with st.sidebar.expander("Dataset", expanded=True):

        file = st.file_uploader("Upload a csv file", type="csv", help=readme["tooltips"]["dataset_upload"])
        
        if file:
            df = pd.read_csv(file)        
        else:
            st.stop()

    # Column names
    with st.sidebar.expander("Columns", expanded=True):
        date_col = st.selectbox(
            "Date column",
            sorted(df.columns)
            if config["columns"]["date"] in ["false", False]
            else [config["columns"]["date"]],
            help=readme["tooltips"]["date_column"],
        )
        target_col = st.selectbox(
            "Target column",
            sorted(set(df.columns) - {date_col})
            if config["columns"]["target"] in ["false", False]
            else [config["columns"]["target"]],
            help=readme["tooltips"]["target_column"],
        )

    st.sidebar.title("2. Modelling")
    
    # External regressors
    with st.sidebar.expander("Regressors"):
        
        if df is not None:
            regressor_cols = st.multiselect(
                "Select external regressors if any",
                list(set(df.columns) - set(["pais","anio_trim","anio","trim","pib_bc_usd","pib_bc_g","pib_bc_des","pib_bc_des_g","pib_bc_int_g","pib_bc_des_int_g","pib_cepal_usd","pib_cepal_des","pib_cepal_g","pib_cepal_des_g","pib_cepal_int_g","pib_cepal_des_int_g"])),
                help=readme["tooltips"]["select_regressors"],
            )

    st.sidebar.title("3. Evaluation")
    # Choose whether or not to do evaluation
    evaluate = st.sidebar.checkbox(
        "Evaluate my model", value=True, help=readme["tooltips"]["choice_eval"]
    )

    if evaluate:

        # Split
        with st.sidebar.expander("Split", expanded=True):
            use_cv = st.checkbox(
                "Perform cross-validation", value=False, help=readme["tooltips"]["choice_cv"]
            )
            col1, col2 = st.columns(2)

            col1.date_input(
               f"Training start date", value=datetime.date(2012, 7, 6), min_value=datetime.date(2012, 7, 6), max_value=datetime.date(2020, 7, 6))

            col2.date_input(
                f"Training end date",
                value=datetime.date(2016, 7, 6),
                max_value=datetime.date(2020, 7, 6),
            )

            col1_val, col2_val = st.columns(2)
            col1_val.date_input(
                "Validation start date",
                value=datetime.date(2020, 7, 6),
                min_value=datetime.date(2020, 7, 6),
                max_value=datetime.date(2025, 7, 6),
            )
            col2_val.date_input(
                "Validation end date",
                value=datetime.date(2023, 7, 6),
                min_value=datetime.date(2023, 7, 6),
                max_value=datetime.date(2025, 7, 6),
            )
            
        # Performance metrics
        with st.sidebar.expander("Metrics", expanded=False):
            eval = input_metrics(readme, config)

        # Scope of evaluation
        with st.sidebar.expander("Scope", expanded=False):
            eval = input_scope_eval(eval, use_cv, readme)
            
    st.sidebar.title("4. Forecast")

if __name__ == "__main__":
    main()
