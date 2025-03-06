import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys

# Parámetros del péndulo
g = 9.81  # gravedad (m/s^2)
L = 1.0   # longitud del péndulo (m)
theta_init = np.pi / 4  # Ángulo inicial (45°)
t_min, t_max = 0, 10  # tiempo de simulación
n_train = 100  # puntos de entrenamiento

# Datos de entrenamiento (tiempo)
t_train = torch.linspace(t_min, t_max, n_train, requires_grad=True).view(-1, 1)

# Definimos la red neuronal
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # Salida: variación de ángulo Δθ
        )

    def forward(self, t):
        return self.net(t) + theta_init  # Se suma el ángulo inicial

# Función de pérdida basada en la ecuación diferencial
def physics_loss(model, t):
    theta = model(t)  # Predicción de θ
    theta_t = torch.autograd.grad(theta, t, grad_outputs=torch.ones_like(theta), create_graph=True)[0]
    theta_tt = torch.autograd.grad(theta_t, t, grad_outputs=torch.ones_like(theta_t), create_graph=True)[0]
    loss_eq = torch.mean((theta_tt + (g / L) * torch.sin(theta))**2)
    
    # Condición inicial (θ(0) = theta_init)
    theta_0_pred = model(torch.tensor([[t_min]], dtype=torch.float32))
    loss_init = torch.mean((theta_0_pred - theta_init) ** 2)
    
    return loss_eq + loss_init

# Inicialización del modelo y optimizador
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entrenamiento
epochs = 5000
loss_history = []
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = physics_loss(model, t_train)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Visualización de la pérdida
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Physics Loss")
plt.show()

# Predicción del ángulo θ en función del tiempo
t_test = torch.linspace(t_min, t_max, 300).view(-1, 1)
theta_pred = model(t_test).detach().numpy()

# Configuración de pygame
pygame.init()

# Dimensiones de la pantalla
WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación de Péndulo - PINN")

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)

# Centro del péndulo
origin = (WIDTH // 2, HEIGHT // 3)
length_px = 200  # Longitud del péndulo en píxeles

# Reloj para controlar la velocidad de la simulación
clock = pygame.time.Clock()

# Simulación en pygame
running = True
i = 0
while running:
    screen.fill(WHITE)  # Fondo

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Obtener el ángulo actual de la predicción
    theta = theta_pred[i % len(theta_pred)][0]  # Obtener ángulo en radianes
    i += 1  # Avanzar en la simulación

    # Calcular la posición del péndulo
    x = origin[0] + length_px * np.sin(theta)
    y = origin[1] + length_px * np.cos(theta)

    # Dibujar el péndulo
    pygame.draw.line(screen, BLACK, origin, (x, y), 4)  # Cuerda
    pygame.draw.circle(screen, RED, (int(x), int(y)), 15)  # Masa del péndulo

    pygame.display.flip()  # Actualizar pantalla
    clock.tick(180)  # Velocidad de actualización (FPS)

pygame.quit()
sys.exit()
