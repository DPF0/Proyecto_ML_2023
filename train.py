# Importar librerías

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
import pickle
import datetime as dt
import os


# Cargar datos

df_train = pd.read_csv('./data/train.csv', index_col= [0])
df_test = pd.read_csv('./data/test.csv', index_col= [0])
df_test_dias = pd.read_csv('./data/test_dias_naturales.csv', index_col= [0])


# Separar X e Y

X_train = df_train.drop(columns= ['€/dia'])
Y_train = df_train['€/dia']

X_test = df_test.drop(columns= ['€/dia'])
Y_test = df_test['€/dia']

X_test_dias = df_test_dias.drop(columns= ['€/dia'])
Y_test_dias = df_test_dias['€/dia']


# Construir pipeline y probar modelos

pipe = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

#Primer modelo del pipeline
random_forest_params = {
    'regressor': [RandomForestRegressor()],
    'regressor__n_estimators': [20, 25, 30, 35, 40, 45, 50],
    'regressor__max_depth': [1, 2, 3]
}                                                                                   

#Segundo modelo del pipeline
linear_regressor_params = {
    'regressor': [LinearRegression()]
}

#Tercer modelo del pipeline
lasso_params = {
    'scaler': [StandardScaler(), MinMaxScaler()],
    'regressor': [Lasso()],
    'regressor__alpha': np.logspace(-4, 3, 100).tolist(),
    'regressor__max_iter': [50000, 100000, 200000]
}

#Cuarto modelo del pipeline
ridge_params = {
    'scaler': [StandardScaler(), MinMaxScaler()],
    'regressor': [Ridge()],
    'regressor__alpha': np.logspace(-4, 3, 100).tolist(),
    'regressor__max_iter': [50000, 100000, 200000]
} 

#Quinto modelo del pipeline
xgb_params = {
    'regressor': [XGBRegressor()],
    'regressor__n_estimators': [20, 25, 30, 35, 40, 45, 50],
    'regressor__max_depth': [1, 2, 3],
    'regressor__learning_rate': [0.01, 0.1, 0.2] 
} 

#Lista de todos los clasificadores con sus parámetros
search_space = [
    random_forest_params, linear_regressor_params,
    lasso_params, ridge_params,
    xgb_params
]


clf = GridSearchCV(estimator = pipe,
                  param_grid = search_space,
                  n_jobs= 6,
                  cv = 10, 
                  scoring= 'neg_mean_absolute_error')

#Se entrena el gridsearch
clf.fit(X_train, Y_train)

# Timestamp
timestamp = dt.datetime.now().strftime('%Y%m%d%H%M%S')

# Salvar el modelo
model = clf.best_estimator_
nombre_modelo = ('./model/modelo_' + timestamp + '.pkl')
pickle.dump(model, open(nombre_modelo, 'wb'))