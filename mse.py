import numpy as np
import matplotlib.pyplot as plt

# Función de pérdida (ejemplo: f(x, y) = 10 - e^-(x^2 + 3y^2))
def loss_function(x, y):
    return 10 - np.exp(- (x**2 + 3*y**2))

# Gradiente de la función de pérdida
def gradient(x, y):
    grad_x = 2 * x * np.exp(- (x**2 + 3*y**2))
    grad_y = 6 * y * np.exp(- (x**2 + 3*y**2))
    return grad_x, grad_y

# Parámetros del descenso de gradiente
learning_rate = 0.1
num_iterations = 1000
x_initial, y_initial = 1, 1  # Valores iniciales para 'x' y 'y'

# Listas para almacenar los valores de 'x', 'y' y la pérdida en cada iteración
x_values = [x_initial]
y_values = [y_initial]
loss_values = [loss_function(x_initial, y_initial)]

# Descenso de gradiente
for i in range(num_iterations):
    gradient_x, gradient_y = gradient(x_initial, y_initial)  # Calcula el gradiente en (x, y)
    x_update = x_initial - learning_rate * gradient_x
    y_update = y_initial - learning_rate * gradient_y
    
    # Almacena los valores de 'x', 'y' y la pérdida
    x_values.append(x_update)
    y_values.append(y_update)
    loss_values.append(loss_function(x_update, y_update))
    
    # Actualiza los valores de 'x' e 'y'
    x_initial, y_initial = x_update, y_update

# Graficar la función de pérdida en 2D
x_range = np.linspace(-1, 1, 100)
y_range = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = loss_function(X, Y)

fig, ax = plt.subplots(figsize=(6, 6))

# Gráfica 2D
ax.contour(X, Y, Z, levels=30, cmap='viridis')
ax.scatter(x_values, y_values, color='red', label='Descenso de Gradiente')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Descenso de Gradiente en el Plano xy')
ax.legend()

plt.show()

print("Resultado final:")
print(f"x = {x_update}, y = {y_update}, Loss = {loss_function(x_update, y_update)}")
