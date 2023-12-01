# Importar librerías
import pickle
import pandas as pd
import numpy as np
import datetime as dt

# Cargar modelo
model = pickle.load(open('./model/modelo_david.pkl', 'rb'))

# Cargar datos
df = pd.read_csv('./data/test.csv', index_col= [0])

# Separar X
X = df.drop(columns= '€/dia')

# Predecir
predictions = pd.DataFrame(model.predict(X))

# Timestamp
timestamp = dt.datetime.now().strftime('%Y%m%d%H%M%S')

# Salvar a csv
predictions.to_csv('./data/predictions '+ timestamp + '.csv')