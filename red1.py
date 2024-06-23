###RED NEURONAL 1


#PASO 1. IMPORTAMOS LAS LIBRERIAS NECESARIAS	
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json

#PASO 2. LEEMOS Y PROCESAMOS EL DATASET
df = pd.read_excel('dataset_completo.xlsx')
X = df.iloc[:, 3:]  #Características (variables de entrada)
y = df['OSAT']  #Variable de salida

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) #Estandarizamos los datos

#PASO 3. DIVIDIMOS EL CONJUNTO DE DATOS EN SUBCONJUNTOS DE ENTRENAMIENTO Y TEST
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


#PASO 4. DEFINIMOS LA ARQUITECTURA DE LA RED NEURONAL
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu',),
    Dense(1, activation='linear')  #Capa de salida para regresión lineal
])

optimizer = Adam(learning_rate=0.001) #Compilación del modelo
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

#PASO 5. ENTRENAMIENTO DE LA RED
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

#PASO 6. PREDICCIONES
test_predictions = model.predict(X_test).flatten()

#PASO 7. MÉTRICAS DE EVALUACIÓN Y GRÁFICOS
mae = mean_absolute_error(y_test, test_predictions)
mse = mean_squared_error(y_test, test_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, test_predictions)
print('ERROR ABSOLUTO MEDIO (MAE): ', mae)
print('ERROR CUADRÁTICO (MSE): ', mse)
print('RAÍZ DEL ERROR CUADRÁTICO MEDIO (RMSE): ', rmse)
print('R2: ', r2)

#Gráfico de dispersión entre los valores reales y los valores predichos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, test_predictions, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='orange', linestyle='--', linewidth=2)
plt.title('Predicciones vs Valores Reales (OSAT)')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.grid(True)
plt.tight_layout()
plt.show()

#Gráficos de métricas por épocas
epochs = range(1, len(history.history['loss']) + 1)
plt.figure(figsize=(14, 8))

#MAE
plt.subplot(2, 2, 1)
plt.plot(epochs, history.history['mae'], label='MAE en entrenamiento')
plt.plot(epochs, history.history['val_mae'], label='MAE en validación')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.legend()
plt.title('MAE por Época')

#MSE
plt.subplot(2, 2, 2)
plt.plot(epochs, history.history['mse'], label='MSE en entrenamiento')
plt.plot(epochs, history.history['val_mse'], label='MSE en validación')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.title('MSE por Época')

#RMSE
plt.subplot(2, 2, 3)
plt.plot(epochs, np.sqrt(history.history['mse']), label='RMSE en entrenamiento')
plt.plot(epochs, np.sqrt(history.history['val_mse']), label='RMSE en validación')
plt.xlabel('Épocas')
plt.ylabel('RMSE')
plt.legend()
plt.title('RMSE por Época')


#R² Score (usando las predicciones para cada época)
r2_scores = []
for epoch in epochs:
    pred_epoch = model.predict(X_train[:epoch])
    r2_scores.append(r2_score(y_train[:epoch], pred_epoch))

plt.subplot(2, 2, 4)
plt.plot(epochs, r2_scores, label='R² Score en entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('R² Score')
plt.legend()
plt.title('R² Score por Época')
plt.tight_layout()
plt.show()

#Pérdida durante el entrenamiento
plt.figure(figsize=(10, 6))
plt.plot(epochs, history.history['loss'], label='Pérdida en entrenamiento')
plt.plot(epochs, history.history['val_loss'], label='Pérdida en validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida de Entrenamiento por Época')
plt.show()

  
