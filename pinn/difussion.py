import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parámetro de difusión térmica
alpha = 0.01

# Generación de datos de entrada
x_f = np.linspace(0, 1, 100)[:, None]  # Dominio espacial
t_f = np.linspace(0, 1, 100)[:, None]  # Dominio temporal
X, T = np.meshgrid(x_f, t_f)
XT_f = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# Condiciones de frontera e inicial
x_ic = np.linspace(0, 1, 100)[:, None]
t_ic = np.zeros_like(x_ic)
u_ic = np.sin(np.pi * x_ic)

x_bc = np.vstack((np.zeros_like(t_f), np.ones_like(t_f)))
t_bc = np.vstack((t_f, t_f))
u_bc = np.zeros_like(x_bc)

# Concatenar datos
XT_ic = np.hstack((x_ic, t_ic))
XT_bc = np.hstack((x_bc, t_bc))
U_ic = u_ic
U_bc = u_bc

# Construcción de la red neuronal
class PINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = [tf.keras.layers.Dense(20, activation='tanh') for _ in range(4)]
        self.out = tf.keras.layers.Dense(1, activation=None)

    def call(self, x):
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)

# Definir la función de pérdida basada en la ecuación de calor
def loss(model, XT_f, XT_ic, XT_bc, U_ic, U_bc):
    with tf.GradientTape(persistent=True) as tape:
        XT_f = tf.convert_to_tensor(XT_f, dtype=tf.float32)
        XT_ic = tf.convert_to_tensor(XT_ic, dtype=tf.float32)
        XT_bc = tf.convert_to_tensor(XT_bc, dtype=tf.float32)

        U_f = model(XT_f)
        U_ic_pred = model(XT_ic)
        U_bc_pred = model(XT_bc)

        # Derivadas parciales
        U_f_t = tape.gradient(U_f, XT_f)[:, 1:2]
        U_f_x = tape.gradient(U_f, XT_f)[:, 0:1]
        U_f_xx = tape.gradient(U_f_x, XT_f)[:, 0:1]

        # Función de pérdida basada en la ecuación de calor
        loss_pde = tf.reduce_mean((U_f_t - alpha * U_f_xx) ** 2)
        loss_ic = tf.reduce_mean((U_ic_pred - U_ic) ** 2)
        loss_bc = tf.reduce_mean((U_bc_pred - U_bc) ** 2)

        return loss_pde + loss_ic + loss_bc

# Entrenamiento
model = PINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

for i in range(5000):
    with tf.GradientTape() as tape:
        loss_value = loss(model, XT_f, XT_ic, XT_bc, U_ic, U_bc)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if i % 500 == 0:
        print(f"Iteración {i}, Pérdida: {loss_value.numpy()}")

# Visualización de resultados
X_test, T_test = np.meshgrid(x_f, t_f)
XT_test = np.hstack((X_test.flatten()[:, None], T_test.flatten()[:, None]))
U_pred = model.predict(XT_test).reshape(100, 100)

plt.imshow(U_pred, extent=[0,1,0,1], origin='lower', aspect='auto', cmap='hot')
plt.colorbar(label='Temperatura')
plt.xlabel("x")
plt.ylabel("t")
plt.title("Solución de la ecuación de calor con PINN")
plt.show()
