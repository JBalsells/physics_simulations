import numpy as np
import matplotlib.pyplot as plt

# Señal de entrada: senoidal + polarización DC
A = 0.01       # Amplitud de la señal (10 mV)
V_bias = 0.75  # Polarización para mantener al transistor en zona activa
f = 1000       # Frecuencia en Hz
t = np.linspace(0, 2e-3, 1000)
vin = V_bias + A * np.sin(2 * np.pi * f * t)

# Parámetros fijos
Vcc = 12
Rc = 1000
N = 5000  # Número de simulaciones Monte Carlo

# Para almacenar resultados
beta_list = []
vc_mean_list = []

# Inicializar figura de 1x3
fig, axs = plt.subplots(1, 3, figsize=(18, 4))

# Graficar la señal de entrada
axs[0].plot(t * 1e3, vin, color='blue')
axs[0].set_title("Entrada: Señal en base (con polarización)")
axs[0].set_xlabel("Tiempo (ms)")
axs[0].set_ylabel("Voltaje (V)")
axs[0].grid(True)

# Simulaciones Monte Carlo de salida
np.random.seed(42)
for _ in range(N):
    beta = np.random.normal(150, 15)
    V_BE = np.random.normal(0.7, 0.02)
    Rb = np.random.normal(1000, 50)

    ib = (vin - V_BE) / Rb
    ic = beta * ib
    vc = Vcc - Rc * ic

    axs[1].plot(t * 1e3, vc, color='orange', alpha=0.2)

    beta_list.append(beta)
    vc_mean_list.append(np.mean(vc))

# Salida en colector
axs[1].set_title(f"Salida: Señales en colector ({N} simulaciones)")
axs[1].set_xlabel("Tiempo (ms)")
axs[1].set_ylabel("Voltaje (V)")
axs[1].grid(True)

# Histograma vc promedio vs beta
axs[2].scatter(beta_list, vc_mean_list, color='purple')
axs[2].set_title("Voltaje medio de colector vs β")
axs[2].set_xlabel("Ganancia β")
axs[2].set_ylabel("Voltaje medio del colector (V)")
axs[2].grid(True)

plt.tight_layout()
plt.show()
