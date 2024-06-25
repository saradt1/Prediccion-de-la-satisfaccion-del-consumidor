#CÓDIGO PARA ANALIZAR LA IMPORTANCIA DE LAS CARACTERÍSTICAS DE ENTRADA SEGÚN CORRELACIÓN E INFORMATION GAIN

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

#Cargamos el dataset
df = pd.read_excel('dataset_completo.xlsx')

#Definimos los datos de entrada y salida
X = df.iloc[:, 3:]  #Valores de entrada
y = df['OSAT']  #Variable de salida

#Coeficiente de Correlación de Pearson
correlations = X.corrwith(y)

#Ganancia de información
info_gain = mutual_info_classif(X, y)

#Almacenamos los resultados en un dataframe
results = pd.DataFrame({
    'Caracteristica': X.columns,
    'Correlacion': correlations.abs(),  
    'Information Gain': info_gain,  
})

#Ordenamos de mayor a menor los datos según la correlacion
results = results.sort_values(by='Correlacion', ascending=False)


#Generamos un gráfico de barras con los resultados
results.set_index('Caracteristica').plot(kind='bar', figsize=(14, 8), width=0.8)
plt.title('Análisis de Selección de Características')
plt.xlabel('Variables de Entrada')
plt.ylabel('Importancia')
plt.legend(loc='upper right')
plt.grid(axis='y')
plt.show()
