import csv
import numpy as np
import matplotlib.pyplot as plt

# Función para cargar datos desde el archivo CSV
def cargar_datos(nombre_archivo):
    datos = []
    with open(nombre_archivo, 'r') as archivo:
        lector = csv.reader(archivo)
        next(lector)
        for fila in lector:
            datos.append([float(fila[1]), float(fila[2])])
    return np.array(datos)

# Función para calcular el error cuadrático medio
def calcular_error(y, y_pred):
    return np.mean((y - y_pred) ** 2)

# Función para realizar el descenso de gradiente
def descenso_gradiente(x, y, b, m, tasa_aprendizaje, epochs):
    n = len(y)

    for _ in range(epochs):
        # Calcular las predicciones
        y_pred = b + m * x

        # Calcular las derivadas parciales
        db = (-2/n) * np.sum(y - y_pred)
        dm = (-2/n) * np.sum(x * (y - y_pred))

        # Actualizar los parámetros usando la tasa de aprendizaje
        b = b - tasa_aprendizaje * db
        m = m - tasa_aprendizaje * dm

    return b, m

# Cargar los datos desde el archivo CSV
datos = cargar_datos('Salary_dataset.csv')

# Separar las columnas en x (característica) y y (etiqueta)
x = datos[:, 0]
y = datos[:, 1]

# Inicializar los parámetros
b_inicial = 0
m_inicial = 0

# Definir la tasa de aprendizaje y el número de epochs
tasa_aprendizaje = 0.01
epochs = 1000

# Realizar el descenso de gradiente
b_optimo, m_optimo = descenso_gradiente(x, y, b_inicial, m_inicial, tasa_aprendizaje, epochs)

# Mostrar los parámetros óptimos
print(f'valor b óptimo: {b_optimo}')
print(f'valor m óptimo: {m_optimo}')

# Graficar la línea de regresión
plt.scatter(x, y, label='Datos reales')
plt.plot(x, b_optimo + m_optimo * x, color='red', label='Línea de regresión con degradiente')
plt.title('Valor de m: ' + str(m_optimo) + '\nValor de b: ' + str(b_optimo))
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
