import streamlit as st
from prophet.utilities import regressor_coefficients

from lib.utils.load import load_config
from lib.inputs.eval import input_metrics, input_scope_eval
import pandas as pd 
import datetime
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from pandas.tseries.offsets import MonthEnd
from stqdm import stqdm

# Page config
#st.set_page_config(page_title="Pronóstico PIB - El Salvador", layout="wide")
# Inject custom CSS to set the width of the sidebar
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 400px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# Load config
config, instructions, readme = load_config(
    "config_streamlit.toml", "config_instructions.toml", "config_readme.toml"
)

def main():

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
            
            year_quarterly_time = list(df.anio_trim.unique())

            with st.sidebar.expander("Training", expanded=True):

                train_init = col1.selectbox(
                    "Training Initial Range",
                    year_quarterly_time, 
                    index = 0
                )

                train_final = col2.selectbox(
                        "Training Final Range",
                        year_quarterly_time[year_quarterly_time.index(train_init)+1:],
                        index = 0
                    )

            with st.sidebar.expander("Test", expanded=True):
                test_init = col1.selectbox(
                        "Test Initial Range",
                        year_quarterly_time[year_quarterly_time.index(train_final)+1:], 
                        index = 0
                    )

                
                test_final = col2.selectbox(
                        "Test Final Range",
                        year_quarterly_time[year_quarterly_time.index(test_init)+1:],
                        index = 0
                    )
        # Performance metrics
        with st.sidebar.expander("Metrics", expanded=False):
            eval = input_metrics(readme, config)

        # Scope of evaluation
        with st.sidebar.expander("Scope", expanded=False):
            eval = input_scope_eval(eval, use_cv, readme)
            
        st.sidebar.title("4. Forecast")
        with st.sidebar.expander("Dataset Regressors", expanded=True):

            file_regressors = st.file_uploader("Upload a csv file for regressors", type="csv", help=readme["tooltips"]["dataset_upload"])
        
            if file_regressors:
                future_reg = pd.read_csv(file_regressors)        
            else:
                st.stop()

        # Scope of evaluation
        with st.sidebar.expander("Horizon", expanded=False):
            horizonte_pronostico = st.selectbox(
                    "Forecast horizon in quarters",
                        range(1,25),
                        index=None,
                        )

    # Info
    with st.expander(
        "App to build a time series forecasting model in a few clicks", expanded=False
    ):
        st.write(readme["app"]["app_intro"])
        st.write("")
    st.write("")

    if st.button("Forecast", use_container_width=True):
        st.markdown("You clicked the Forecast button.")


        df["date"] = pd.PeriodIndex(df[date_col].str.replace(" ","-"), freq='Q')
        df["ds"] = pd.PeriodIndex(df[date_col].str.replace(" ","-"), freq='Q').to_timestamp()

        df.set_index("date", inplace = True)

        ### Define train init-final y test init-final
        ### Define train periods

        year_train_init, quarter_train_init = train_init.split()
        year_train_final, quarter_train_final = train_final.split()

        year_test_init, quarter_test_init = test_init.split()
        year_test_final, quarter_test_final = test_final.split()

        year_train_init, year_train_final, year_test_init, year_test_final = int(year_train_init), int(year_train_final), int(year_test_init), int(year_test_final) 

        quarter_train_init, quarter_train_final, quarter_test_init, quarter_test_final = int(quarter_train_init.replace("Q","")), int(quarter_train_final.replace("Q","")), int(quarter_test_init.replace("Q","")), int(quarter_test_final.replace("Q",""))


        ### Define test periods

        train_init = pd.Timestamp(year_train_init, quarter_train_init*3, 1)
        train_final = pd.Timestamp(year_train_final, quarter_train_final*3, 1) + MonthEnd(1)

        test_init = pd.Timestamp(year_test_init, quarter_test_init*3, 1)
        test_final = pd.Timestamp(year_test_final, quarter_test_final*3, 1)+ MonthEnd(1)

        train_range = pd.PeriodIndex(pd.date_range(start=train_init, end=train_final, freq="QE"), freq = "Q")
        test_range = pd.PeriodIndex(pd.date_range(start=test_init, end=test_final, freq="QE"), freq = "Q")

        ### Get data

        if regressor_cols:
            data_input = df[["ds", target_col]+regressor_cols].rename(columns = {target_col : "y"})
        else:
            data_input = df[["ds", target_col]].rename(columns = {target_col : "y"})

        data_input["y"] =  np.log(data_input["y"])

        train = data_input.loc[train_range]
        test = data_input.loc[test_range]


        from itertools import product

        param_grid = {
            #'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            #'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
            'changepoint_prior_scale': [0.001, 0.01],
            'seasonality_prior_scale': [0.01, 0.1]
        }

        params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]


        mses = []
        with st.spinner("Training the model...", show_time=True):

            for param in stqdm(params, desc="This is a slow task", mininterval=1):
                m = Prophet(**param)

                if regressor_cols:
                    for i in regressor_cols:
                        m.add_regressor(i)

                m.add_country_holidays(country_name='US')
                m.fit(train)
                
                #df_cv = cross_validation(model=m, horizon='365 days') # Original
                df_cv = cross_validation(model=m, horizon='730 days')
                #df_cv = cross_validation(model=m, initial='1460 days', period='365 days', horizon='365 days')
                df_p = performance_metrics(df_cv, rolling_window=1)
                mses.append(df_p['mse'].values[0])

        st.success("Done!")

        tuning_results = pd.DataFrame(params)
        tuning_results['mse'] = mses



        best_params = params[np.argmin(mses)]
        print(best_params)



        m = Prophet(**best_params)
        m.add_country_holidays(country_name='SL')

        if regressor_cols:
            for i in regressor_cols:
                m.add_regressor(i)
                
        m.fit(train)

        future = m.make_future_dataframe(periods = test.shape[0] + 1 + horizonte_pronostico, freq='Q')

        future["date"] = pd.PeriodIndex(future["ds"], freq = "Q")
        future = future.set_index("date")

        all_time_range = pd.PeriodIndex(pd.date_range(start=train_init, end=test_final, freq="QE"), freq = "Q")

        future = pd.concat([future, df.loc[all_time_range, regressor_cols]], axis = 1)

        future_without_reg = future.ffill()

        forecast_without_regs = m.predict(future_without_reg)

        ## Agrega regresores actualizados
        future_with_reg = future.copy()
        fechas_sin_dato = future[future.isna().any(axis =1)].ds

        future_reg["date"] = pd.PeriodIndex(future_reg[date_col].str.replace(" ","-"), freq='Q')
        future_reg["ds"] = pd.PeriodIndex(future_reg[date_col].str.replace(" ","-"), freq='Q').to_timestamp()

        future_reg.set_index("date", inplace = True)
        
        #st.dataframe(future_with_reg)

        #st.dataframe(future_reg)

        future_with_reg.loc[future_reg.index,regressor_cols] = future_reg

        future_with_reg = future_with_reg.ffill()

        forecast_with_regs = m.predict(future_with_reg)

        ## Visualicemos desempeño
        all_dataset = pd.concat([train,test])[["ds", "y"]]
        all_dataset["yhat"] = all_dataset["y"]
        all_dataset["yhat_upper"] = all_dataset["y"]
        all_dataset["yhat_lower"] = all_dataset["y"]


        ## Prepare forecasts
        forecast_without_regs["date"] = pd.PeriodIndex(forecast_without_regs["ds"], freq = "Q")
        forecast_with_regs["date"] = pd.PeriodIndex(forecast_with_regs["ds"], freq = "Q")

        forecast_without_regs.set_index("date", inplace = True)
        forecast_with_regs.set_index("date", inplace = True)

        #forecast_with_regs.loc[fechas_sin_dato.index]

        ## Concat data
        all_dataset = pd.concat([all_dataset, forecast_with_regs.loc[fechas_sin_dato.index][["ds", "yhat", "yhat_lower", "yhat_upper"]]])

        #all_dataset[["ds", "yhat", "yhat_lower", "yhat_upper"]] = all_dataset[["ds", "yhat", "yhat_lower", "yhat_upper"]].apply(lambda x : np.exp(x))

        import plotly.graph_objects as go

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            name="Pronóstico",
            x=all_dataset["ds"], 
            y=all_dataset["yhat"],
            mode='lines',
            line_color='red',
            ))


        fig.add_trace(
            go.Scatter(
                name = "Serie Observada",
                x=all_dataset["ds"],
                y=all_dataset["y"],
                mode='lines', line_color='darkblue'
            )
        )

        fig.add_trace(go.Scatter(
            showlegend=False,
            x=all_dataset.iloc[-(len(fechas_sin_dato.index)+1):]["ds"], 
            y=all_dataset.iloc[-(len(fechas_sin_dato.index)+1):]["yhat_upper"],
            fill=None,
            mode='lines',
            line_color='white',
            ))
        fig.add_trace(go.Scatter(
            showlegend=False,
            x=all_dataset.iloc[-(len(fechas_sin_dato.index)+1):]["ds"],
            y=all_dataset.iloc[-(len(fechas_sin_dato.index)+1):]["yhat_lower"],
            fill='tonexty', fillcolor='rgba(255, 255, 0, 0.5)', line_color='white',# fill area between trace0 and trace1
            mode='lines'))



        fig.update_layout(
            height=800,
            title=dict(text='Pronóstico del Modelo', font=dict(size=30)),

            xaxis=dict(
                title=dict(
                    text="Tiempo"
                ),
            ),
            yaxis=dict(
                title=dict(
                    text="Pronóstico"
                )
            ),
            legend=dict(
                title=dict(
                    text="Grupo"
                )
            ),
            font=dict(
                size=20,
            )
        )
        # Plot!
        st.plotly_chart(fig)

        st.dataframe(all_dataset)
        st.dataframe(regressor_coefficients(m))

if __name__ == "__main__":
    main()
