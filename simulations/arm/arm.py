import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Longitudes de los eslabones del brazo robótico
L1, L2, L3, L4, L5, L6 = 50, 50, 50, 50, 50, 50

def forward_kinematics(theta):
    """ Calcula la posición (x, y, z) del efector final."""
    t1, t2, t3, t4, t5, t6 = np.radians(theta)
    
    x1 = L1 * np.cos(t1)
    y1 = L1 * np.sin(t1)
    z1 = 0
    
    x2 = x1 + L2 * np.cos(t1) * np.cos(t2)
    y2 = y1 + L2 * np.sin(t1) * np.cos(t2)
    z2 = z1 + L2 * np.sin(t2)
    
    x3 = x2 + L3 * np.cos(t1) * np.cos(t2 + t3)
    y3 = y2 + L3 * np.sin(t1) * np.cos(t2 + t3)
    z3 = z2 + L3 * np.sin(t2 + t3)
    
    x4 = x3 + L4 * np.cos(t1) * np.cos(t2 + t3 + t4)
    y4 = y3 + L4 * np.sin(t1) * np.cos(t2 + t3 + t4)
    z4 = z3 + L4 * np.sin(t2 + t3 + t4)
    
    x5 = x4 + L5 * np.cos(t1) * np.cos(t2 + t3 + t4 + t5)
    y5 = y4 + L5 * np.sin(t1) * np.cos(t2 + t3 + t4 + t5)
    z5 = z4 + L5 * np.sin(t2 + t3 + t4 + t5)
    
    x6 = x5 + L6 * np.cos(t1) * np.cos(t2 + t3 + t4 + t5 + t6)
    y6 = y5 + L6 * np.sin(t1) * np.cos(t2 + t3 + t4 + t5 + t6)
    z6 = z5 + L6 * np.sin(t2 + t3 + t4 + t5 + t6)
    
    return np.array([[0, 0, 0], [x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4], [x5, y5, z5], [x6, y6, z6]])

# Inicializar la ventana de Matplotlib en modo interactivo
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def plot_arm(joints):
    ax.clear()  # Limpiar la figura antes de graficar el nuevo estado
    ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], marker='o', color='black')
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_zlim([0, 200])
    plt.draw()
    plt.pause(0.05)  # Pequeña pausa para permitir la actualización de la ventana

# Posición inicial y final
initial_theta = [0, 45, -30, 20, 15, 10]
target_theta = [30, -20, 40, -10, -15, 5]

# Generar trayectorias
steps = 50
theta_traj = np.linspace(initial_theta, target_theta, steps)

for theta in theta_traj:
    joints = forward_kinematics(theta)
    plot_arm(joints)

plt.ioff()  # Desactivar modo interactivo al finalizar
plt.show()  # Mantener la ventana abierta al terminar
