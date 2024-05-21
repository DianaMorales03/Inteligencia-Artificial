# Importar la biblioteca numpy para manejar matrices y operaciones vectoriales
import numpy as np

# Definir los pesos y el umbral del perceptrón
weights = np.array([1, 1])
bias = -1.5

# Definir la función de activación
def activation_function(inputs):
    # Calcular la suma ponderada de las entradas y el sesgo
    weighted_sum = np.dot(weights, inputs) + bias

    # Aplicar la función de activación (en este caso, una función escalón)
    return 1 if weighted_sum >= 0 else 0

# Definir los datos de ejemplo
data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

# Probar el perceptrón con los datos de ejemplo
for inputs, expected_output in data:
    inputs_array = np.array(inputs)
    output = activation_function(inputs_array)
    print(f"Entradas: {inputs}\nSalida esperada: {expected_output}\nSalida del perceptrón: {output}\n")
