import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_excel('dataset.xlsx')

# Procesar las columnas seleccionadas
selected_columns = df.iloc[:, 3:]
column_means = selected_columns.mean()  # Calcular la media de cada columna para reemplazar los valores faltantes
selected_columns = selected_columns.fillna(column_means)  # Rellenar con la media de cada columna
selected_columns = selected_columns.astype(float)  # Conversión a valores flotantes si los datos originales no son enteros
df_processed = pd.concat([df.iloc[:, :3], selected_columns], axis=1)  # Combinar las columnas originales con las procesadas

# Definir datos de entrada y salida
X = df_processed.iloc[:, 3:]  # Valores de entrada
y = df_processed['OSAT']  # Variable de salida

#Correlación de Pearson
correlations = X.corrwith(y)

#Ganancia de información
info_gain = mutual_info_classif(X, y)

#Clasificador Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importances = rf.feature_importances_

"""#DataFrame con las importancias normalizadas
results = pd.DataFrame({
    'Caracteristica': X.columns,
    'Correlacion': correlations.abs(),  #Valor absoluto de la correlación
    'Information Gain': info_gain / info_gain.max(),  #Normalizar information gain
    'Random Forest': feature_importances / feature_importances.max()  #Normalizar random forest
})

#DataFrame ordenado de mayor a menos ganancia de información
results = results.sort_values(by='Information Gain', ascending=False)"""

# DataFrame sin normalizar
results = pd.DataFrame({
    'Caracteristica': X.columns,
    'Correlacion': correlations.abs(),  # Valor absoluto de la correlación
    'Information Gain': info_gain,  # Sin normalizar information gain
    'Random Forest': feature_importances  # Sin normalizar random forest
})

# DataFrame ordenado de mayor a menos ganancia de información
results = results.sort_values(by='Information Gain', ascending=False)

#Imprimir gáfico de importancia
results.set_index('Caracteristica').plot(kind='bar', figsize=(14, 8), width=0.8)
plt.title('Análisis para la selección de Variables de Entrada')
plt.xlabel('Variables de Entrada')
plt.ylabel('Importancia de cada variable de entrada (sin normalizar)')
plt.legend(loc='upper right')
plt.grid(axis='y')
plt.show()
