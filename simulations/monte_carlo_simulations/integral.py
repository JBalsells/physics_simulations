import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(x):
    return x**2

def F(x):
    return (x**3) / 3

a, b = 0, 4
N = 40000

x_random = np.random.uniform(a, b, N)
f_random = f(x_random)
mc_integral = (b - a) * np.mean(f_random)
real_integral, error = quad(f, a, b)

print(f"Estimación Monte Carlo: {mc_integral:.6f}")
print(f"Integral real (scipy):  {real_integral:.6f}")
print(f"Error estimado (scipy): {error:.2e}")
print(f"Error absoluto Monte Carlo: {abs(mc_integral - real_integral):.6f}")

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. f(x)
x_vals = np.linspace(a, b, 500)
y_vals = f(x_vals)
axs[0, 0].plot(x_vals, y_vals, label=r"$f(x) = x^2$", color='blue')
axs[0, 0].fill_between(x_vals, y_vals, alpha=0.2, color='blue')
axs[0, 0].set_title("Función $f(x) = x^2$")
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("f(x)")
axs[0, 0].grid(True)
axs[0, 0].legend()

# 2. Histograma de x_random
axs[0, 1].hist(x_random, bins=500, density=True, alpha=0.7, color='orange')
axs[0, 1].set_title("Histograma de $x$ ~ Uniforme[0, 4]")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("Densidad")
axs[0, 1].grid(True)

# 3. Histograma de f(x)
axs[1, 0].hist(f_random, bins=100, density=True, alpha=0.7, color='green')
axs[1, 0].axvline(np.mean(f_random), color='red', linestyle='--', label='Media de f(x)')
axs[1, 0].set_title("Histograma de $f(x) = x^2$")
axs[1, 0].set_xlabel("f(x)")
axs[1, 0].set_ylabel("Frecuencia")
axs[1, 0].legend()
axs[1, 0].grid(True)

# 4. Comparación entre primitiva y puntos Monte Carlo
axs[1, 1].plot(x_vals, f(x_vals), label=r"$f(x) = x^2$", color='blue')
axs[1, 1].scatter(x_random[:1000], f_random[:1000], alpha=0.2, color='red', s=100, label="Puntos Monte Carlo")
axs[1, 1].set_title("Área bajo la curva vs. puntos Monte Carlo")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("f(x)")
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
