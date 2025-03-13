import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros de la simulación
N = 100  # Tamaño de la cuadrícula (NxN)
dx = 1.0  # Paso espacial
dt = 0.5  # Paso temporal
c = 1.2   # Velocidad de propagación de la onda
T = 200   # Número de pasos de tiempo

# Factores de la ecuación de onda
r = (c * dt / dx) ** 2

# Inicialización de la onda en el tiempo
u = np.zeros((N, N))  # Estado actual
u_prev = np.zeros((N, N))  # Estado en el paso anterior
u_next = np.zeros((N, N))  # Estado en el siguiente paso

# Condición inicial: un pulso en el centro
u[N//2, N//2] = 1.0

# Función para actualizar la onda con Runge-Kutta
def update_wave():
    global u, u_prev, u_next
    for i in range(1, N-1):
        for j in range(1, N-1):
            # Método de diferencias finitas para la ecuación de onda
            laplacian = (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4*u[i, j])
            k1 = dt**2 * c**2 * laplacian
            k2 = dt**2 * c**2 * (laplacian + k1 / 2)
            k3 = dt**2 * c**2 * (laplacian + k2 / 2)
            k4 = dt**2 * c**2 * (laplacian + k3)
            
            # Runge-Kutta para actualizar la onda
            u_next[i, j] = 2*u[i, j] - u_prev[i, j] + (k1 + 2*k2 + 2*k3 + k4) / 6

    # Aplicamos condiciones de frontera (bordes fijos u=0)
    u_next[0, :] = u_next[-1, :] = 0
    u_next[:, 0] = u_next[:, -1] = 0

    # Actualizamos los estados
    u_prev[:, :] = u
    u[:, :] = u_next

fig, ax = plt.subplots()
im = ax.imshow(u, cmap='viridis', vmin=-1, vmax=2)

def animate(frame):
    update_wave()
    im.set_array(u)
    return [im]

ani = animation.FuncAnimation(fig, animate, frames=T, interval=50, blit=True)
plt.show()
