import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation
import matplotlib.pyplot as plt
from prophet.diagnostics import performance_metrics
from pandas.tseries.offsets import MonthEnd

from sklearn.impute import KNNImputer

df = pd.read_csv("data/datos.csv")


df["date"] = pd.PeriodIndex(df["anio_trim"].str.replace(" ","-"), freq='Q')
df["ds"] = pd.PeriodIndex(df["anio_trim"].str.replace(" ","-"), freq='Q').to_timestamp()

df.set_index("date", inplace = True)

### Define train periods
year_train_init = 2010
year_train_final = 2023
quarter_train_init = 1
quarter_train_final = 1

### Define test periods
year_test_init = 2023
year_test_final = 2024
quarter_test_init = 2
quarter_test_final = 3

### Define train init-final y test init-final
train_init = pd.Timestamp(year_train_init, quarter_train_init*3, 1)
train_final = pd.Timestamp(year_train_final, quarter_train_final*3, 1)+ MonthEnd(1)

test_init = pd.Timestamp(year_test_init, quarter_test_init*3, 1)
test_final = pd.Timestamp(year_test_final, quarter_test_final*3, 1)+ MonthEnd(1)

train_range = pd.PeriodIndex(pd.date_range(start=train_init, end=train_final, freq="QE"), freq = "Q")
test_range = pd.PeriodIndex(pd.date_range(start=test_init, end=test_final, freq="QE"), freq = "Q")

### Get data
target_var = "pib_bc_usd"

data_input = df[["ds", target_var]].rename(columns = {target_var : "y"})
data_input["y"] =  np.log(data_input["y"])

train = data_input.loc[train_range]
test = data_input.loc[test_range]

### Train model
m = Prophet()
m.fit(train)

future = m.make_future_dataframe(periods=2)
forecast = m.predict(future)

fig1 = m.plot(forecast)
plt.show()

######### MONTHLY CHOCOLATE

df = pd.read_csv('data/monthly_chocolate_search_usa.csv')



fig, ax = plt.subplots()

ax.plot(df['chocolate'])
ax.set_xlabel('Date')
ax.set_ylabel('Proportion of searches using with the keyword "chocolate"')

plt.xticks(np.arange(0, 215, 12), np.arange(2004, 2022, 1))

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

df.columns = ['ds', 'y']

from pandas.tseries.offsets import MonthEnd

df['ds'] = pd.to_datetime(df['ds']) + MonthEnd(1)

train = df[:-12]
test = df[-12:]



from itertools import product

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
}

params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

mses = []

cutoffs = pd.date_range(start='2009-01-31', end='2020-01-31', freq='12M')

for param in params:
    m = Prophet(**param)
    m.add_country_holidays(country_name='US')
    m.fit(train)
    
    df_cv = cross_validation(model=m, horizon='365 days')
    df_p = performance_metrics(df_cv, rolling_window=1)
    mses.append(df_p['mse'].values[0])
    
tuning_results = pd.DataFrame(params)
tuning_results['mse'] = mses



best_params = params[np.argmin(mses)]
print(best_params)



m = Prophet(**best_params)
m.add_country_holidays(country_name='US')
m.fit(train)

future = m.make_future_dataframe(periods=12, freq='M')
forecast = m.predict(future)

fig1 = m.plot(forecast)
plt.show()


### GDP
from pandas.tseries.offsets import QuarterEnd

df = pd.read_csv("data/datos.csv")

df["date"] = pd.PeriodIndex(df["anio_trim"].str.replace(" ","-"), freq='Q')
df["ds"] = pd.PeriodIndex(df["anio_trim"].str.replace(" ","-"), freq='Q').to_timestamp()


df.set_index("date", inplace = True)

### Get data
target_var = "pib_bc_usd"

data_input = df[["ds", target_var]].rename(columns = {target_var : "y"})
data_input["y"] =  np.log(data_input["y"])

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


def task(param):
    
    m = Prophet(**param)
    m.add_country_holidays(country_name='US')
    m.fit(train)
    
    df_cv = cross_validation(model=m, horizon='365 days')
    df_p = performance_metrics(df_cv, rolling_window=1)
    return df_p['mse'].values[0]

from multiprocessing import Pool

p = Pool(processes = 10)
mses = p.map(task, params)
    
tuning_results = pd.DataFrame(params)
tuning_results['mse'] = mses



best_params = params[np.argmin(mses)]
print(best_params)



m = Prophet(**best_params)
m.add_country_holidays(country_name='US')
m.fit(train)

future = m.make_future_dataframe(periods=12, freq='Q')
forecast = m.predict(future)

fig1 = m.plot(forecast)
plt.show()


### GDP Más regresores
imputer = KNNImputer(n_neighbors=2, weights="uniform")

df = pd.read_csv("data/datos.csv")

df["date"] = pd.PeriodIndex(df["anio_trim"].str.replace(" ","-"), freq='Q')
df["ds"] = pd.PeriodIndex(df["anio_trim"].str.replace(" ","-"), freq='Q').to_timestamp()


df.set_index("date", inplace = True)

### Get data
target_var = "pib_bc_usd"
aditional_vars = ["remesas_usd_trim"]

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


def task(param):
    
    m = Prophet(**param)

    for i in aditional_vars:
        m.add_regressor(i)

    m.add_country_holidays(country_name='US')
    m.fit(train)
    
    #df_cv = cross_validation(model=m, horizon='365 days') # Original
    df_cv = cross_validation(model=m, horizon='730 days')
    #df_cv = cross_validation(model=m, initial='1460 days', period='365 days', horizon='365 days')
    df_p = performance_metrics(df_cv, rolling_window=1)
    return df_p['mse'].values[0]

from multiprocessing import Pool

p = Pool(processes = 16)
mses = p.map(task, params)
    
tuning_results = pd.DataFrame(params)
tuning_results['mse'] = mses



best_params = params[np.argmin(mses)]
print(best_params)



m = Prophet(**best_params)
m.add_country_holidays(country_name='US')
m.fit(train)

future = m.make_future_dataframe(periods=12, freq='Q')

for i in aditional_vars:
    future[i] = df.loc[train_range[0]:,i].reset_index(drop=True)


forecast = m.predict(future)

fig1 = m.plot(forecast)
plt.show()



fig, ax = plt.subplots()
ax.plot(forecast["ds"], forecast["yhat"], '-')
ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2)
ax.plot(train["ds"], train["y"], 'o', color='tab:brown')
ax.plot(test["ds"], test["y"], 'o', color='tab:red')

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    showlegend=False,
    x=forecast["ds"], 
    y=forecast["yhat_upper"],
    fill=None,
    mode='lines',
    line_color='white',
    ))
fig.add_trace(go.Scatter(
    showlegend=False,
    x=forecast["ds"],
    y=forecast["yhat_lower"],
    fill='tonexty', fillcolor='rgba(255, 255, 0, 0.5)', line_color='white',# fill area between trace0 and trace1
    mode='lines'))

fig.add_trace(go.Scatter(
    name = "Pronóstico",
    x=forecast["ds"],
    y=forecast["yhat"],
    mode='lines', line_color='darkblue'))

fig.add_trace(go.Scatter(
    name = "Datos Entrenamiento",
    x=train["ds"],
    y=train["y"],
    mode='markers', line_color='black'))

fig.add_trace(go.Scatter(
    name = "Datos Prueba",
    x=test["ds"],
    y=test["y"],
    mode='markers', line_color='red'))

fig.update_traces(textposition='top center',
                    marker={'size': 12, "line" : {"width" : 3, "color" : 'DarkSlateGrey'}}
                  )

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


### Carga datos
df = pd.read_csv("data/datos.csv")

df["date"] = pd.PeriodIndex(df["anio_trim"].str.replace(" ","-"), freq='Q')
df["ds"] = pd.PeriodIndex(df["anio_trim"].str.replace(" ","-"), freq='Q').to_timestamp()


df.set_index("date", inplace = True)

### Get data
target_var = "pib_bc_usd"
aditional_vars = ["remesas_usd_trim"]

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

tuning_results = pd.DataFrame(params)
tuning_results['mse'] = mses



best_params = params[np.argmin(mses)]
print(best_params)



m = Prophet(**best_params)
m.add_country_holidays(country_name='US')
m.fit(train)

future = m.make_future_dataframe(periods=12, freq='Q')

for i in aditional_vars:
    future[i] = df.loc[train_range[0]:,i].reset_index(drop=True)


forecast = m.predict(future)


