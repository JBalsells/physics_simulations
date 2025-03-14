import numpy as np
import matplotlib.pyplot as plt

# dy/dx = f(x, y)
def f(x, y):
    return x + y

def runge_kutta_4(f, x0, y0, h, n):
    x = x0
    y = y0
    xs = [x]
    ys = [y]
    
    for _ in range(n):
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        
        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h
        
        xs.append(x)
        ys.append(y)
    
    return np.array(xs), np.array(ys)

x0, y0 = 0, 1  # Condición inicial: y(0) = 1
h = 0.1  # paso
x_max = 2  # limite superior
n = int((x_max - x0) / h)  # pasos

# Resolucion
x_vals, y_vals = runge_kutta_4(f, x0, y0, h, n)

plt.plot(x_vals, y_vals, 'bo-', label="Solución Numérica (RK4)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Solución de dy/dx = x + y con Runge-Kutta de cuarto orden")
plt.grid()
plt.show()