import numpy as np
import matplotlib.pyplot as plt

# Número de simulaciones
N = 100000
v0_mean = 50.0  # Velocidad inicial media en m/s

# Rango de distancia objetivo en metros
d_min = 50
d_max = 100

# Generación de variables aleatorias
v0 = np.random.normal(v0_mean, 0.5, N)  # Velocidad inicial con ±5% de variación
g = np.random.normal(9.81, 0.01, N)      # Gravedad con ligera variación
angles_deg = np.random.uniform(0, 90, N)  # Ángulos en grados entre 0 y 90
angles_rad = np.radians(angles_deg)       # Conversión a radianes

# Cálculo del alcance para cada combinación
ranges = (v0 ** 2) * np.sin(2 * angles_rad) / g

# Filtrado de ángulos que cumplen con el rango de distancia deseado
valid_angles = angles_deg[(ranges >= d_min) & (ranges <= d_max)]

# Impresión de resultados
if len(valid_angles) == 0:
    print("No se encontró ningún ángulo que cumpla con el rango especificado.")
else:
    angle_min = np.min(valid_angles)
    angle_max = np.max(valid_angles)
    print(f"Ángulo mínimo válido: {angle_min:.2f}°")
    print(f"Ángulo máximo válido: {angle_max:.2f}°")
    print(f"Número de casos válidos: {len(valid_angles)}")

# Creación de subgráficos extendido a 3 columnas x 2 filas
fig, axs = plt.subplots(2, 3, figsize=(18, 8))

# Histograma de la velocidad inicial v0
axs[0, 0].hist(v0, bins=100, color='skyblue', edgecolor='black')
axs[0, 0].set_title("Distribución de la velocidad inicial $v_0$")
axs[0, 0].set_xlabel("Velocidad (m/s)")
axs[0, 0].set_ylabel("Frecuencia")
axs[0, 0].grid(True)

# Histograma de la gravedad g
axs[0, 1].hist(g, bins=100, color='lightcoral', edgecolor='black')
axs[0, 1].set_title("Distribución de la gravedad $g$")
axs[0, 1].set_xlabel("Gravedad (m/s²)")
axs[0, 1].set_ylabel("Frecuencia")
axs[0, 1].grid(True)

# Histograma del alcance (range)
axs[0, 2].hist(ranges, bins=100, color='goldenrod', edgecolor='black')
axs[0, 2].set_title("Distribución del alcance $R$")
axs[0, 2].set_xlabel("Alcance (m)")
axs[0, 2].set_ylabel("Frecuencia")
axs[0, 2].grid(True)

# Histograma de todos los ángulos generados
axs[1, 0].hist(angles_deg, bins=100, color='lightgray', edgecolor='black')
axs[1, 0].set_title("Distribución de todos los ángulos generados")
axs[1, 0].set_xlabel("Ángulo (°)")
axs[1, 0].set_ylabel("Frecuencia")
axs[1, 0].grid(True)

# Histograma de los ángulos válidos
axs[1, 1].hist(valid_angles, bins=100, color='lightgreen', edgecolor='black')
axs[1, 1].set_title("Ángulos que cumplen con el rango objetivo")
axs[1, 1].set_xlabel("Ángulo (°)")
axs[1, 1].set_ylabel("Frecuencia")
axs[1, 1].grid(True)

# Espacio vacío para que el layout se ajuste mejor
axs[1, 2].axis('off')

# Ajuste del diseño y visualización
plt.tight_layout()
plt.show()
