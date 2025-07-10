import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation
import matplotlib.pyplot as plt
from prophet.diagnostics import performance_metrics
from pandas.tseries.offsets import MonthEnd

from sklearn.impute import KNNImputer
from prophet.utilities import regressor_coefficients

### Test GDP agregando regresores recientes
## Define imputer
imputer = KNNImputer(n_neighbors=2, weights="uniform")

### Define train periods
year_train_init = 2010
year_train_final = 2023
quarter_train_init = 1
quarter_train_final = 1

### Define test periods
year_test_init = 2023
year_test_final = 2024
quarter_test_init = 2
quarter_test_final = 2

### Define train init-final y test init-final
train_init = pd.Timestamp(year_train_init, quarter_train_init*3, 1)
train_final = pd.Timestamp(year_train_final, quarter_train_final*3, 1)+ MonthEnd(1)

test_init = pd.Timestamp(year_test_init, quarter_test_init*3, 1)
test_final = pd.Timestamp(year_test_final, quarter_test_final*3, 1)+ MonthEnd(1)

train_range = pd.PeriodIndex(pd.date_range(start=train_init, end=train_final, freq="QE"), freq = "Q")
test_range = pd.PeriodIndex(pd.date_range(start=test_init, end=test_final, freq="QE"), freq = "Q")
all_time_range = pd.PeriodIndex(pd.date_range(start=train_init, end=test_final, freq="QE"), freq = "Q")


### Carga datos
df = pd.read_csv("data/datos.csv")

df["date"] = pd.PeriodIndex(df["anio_trim"].str.replace(" ","-"), freq='Q')
df["ds"] = pd.PeriodIndex(df["anio_trim"].str.replace(" ","-"), freq='Q').to_timestamp()


df.set_index("date", inplace = True)

### Get data
target_var = "pib_bc_usd"
aditional_vars = ["remesas_usd_trim", "consumo_elect_total"]

data_input = df[["ds", target_var]+aditional_vars].rename(columns = {target_var : "y"})
data_input["y"] =  np.log(data_input["y"])

data_input[["y"] + aditional_vars] = imputer.fit_transform(data_input.set_index("ds"))

train = data_input.loc[train_range]
test = data_input.loc[test_range]


from itertools import product

param_grid = {
    #'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    #'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
    'changepoint_prior_scale': [0.001, 0.01, 0.1],
    'seasonality_prior_scale': [0.01, 0.1, 1.0]
}

params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]


mses = []

for param in params:
    m = Prophet(**param)

    if aditional_vars:
        for i in aditional_vars:
            m.add_regressor(i)

    m.add_country_holidays(country_name='SL')
    m.fit(train)
    
    #df_cv = cross_validation(model=m, horizon='365 days') # Original
    df_cv = cross_validation(model=m, horizon='730 days')
    #df_cv = cross_validation(model=m, initial='1460 days', period='365 days', horizon='365 days')
    df_p = performance_metrics(df_cv, rolling_window=1)
    mses.append(df_p['mse'].values[0])

tuning_results = pd.DataFrame(params)
tuning_results['mse'] = mses



best_params = params[np.argmin(mses)]
print(best_params)



m = Prophet(**best_params)
m.add_country_holidays(country_name='SL')

if aditional_vars:
    for i in aditional_vars:
        m.add_regressor(i)
        
m.fit(train)

regressor_coefficients(m)

forecast_quarters = 1

future = m.make_future_dataframe(periods = test.shape[0] + 1 + forecast_quarters, freq='Q')

future["date"] = pd.PeriodIndex(future["ds"], freq = "Q")
future = future.set_index("date")


future = pd.concat([future, df.loc[all_time_range, aditional_vars]], axis = 1)

future_without_reg = future.ffill()

forecast_without_regs = m.predict(future_without_reg)

## Agrega regresores actualizados
future_with_reg = future.copy()
fechas_sin_dato = future[future.isna().any(axis =1)].ds

#future_reg = df.loc[fechas_sin_dato.index,aditional_vars ]
index_on_data = list(set(df.index).intersection(fechas_sin_dato.index))
future_reg = df.loc[index_on_data,aditional_vars ]


future_with_reg.loc[future_reg.index,aditional_vars] = future_reg

future_with_reg = future_with_reg.ffill()

forecast_with_regs = m.predict(future_with_reg)

forecast_without_regs[["ds", "yhat"]].iloc[-1, :]
forecast_with_regs[["ds", "yhat"]].iloc[-1, :]

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

forecast_with_regs.loc[fechas_sin_dato.index]

## Concat data
all_dataset = pd.concat([all_dataset, forecast_with_regs.loc[fechas_sin_dato.index][["ds", "yhat", "yhat_lower", "yhat_upper"]]])

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
fig.show()
