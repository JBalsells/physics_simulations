"""
Author: Jorge A. Balsells Orellana
Date: March 6 2025
Description: Physics-Informed Neural Network (PINN) for physics simulatinons.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys

# Pendulum parameters
g = 9.81  # gravity (m/s^2)
L = 1.0   # pendulum length (m)
theta_init = np.pi / 4  # initial angle (45°)
t_min, t_max = 0, 10  # simulation time
n_train = 100  # training points

# Training data (time)
t_train = torch.linspace(t_min, t_max, n_train, requires_grad=True).view(-1, 1)

# Define the neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # Output: angular variation Δθ
        )

    def forward(self, t):
        return self.net(t) + theta_init  # Add the initial angle

# Physics-based loss function
def physics_loss(model, t):
    theta = model(t)  # θ prediction
    theta_t = torch.autograd.grad(theta, t, grad_outputs=torch.ones_like(theta), create_graph=True)[0]
    theta_tt = torch.autograd.grad(theta_t, t, grad_outputs=torch.ones_like(theta_t), create_graph=True)[0]
    loss_eq = torch.mean((theta_tt + (g / L) * torch.sin(theta))**2)
    
    # Initial condition (θ(0) = theta_init)
    theta_0_pred = model(torch.tensor([[t_min]], dtype=torch.float32))
    loss_init = torch.mean((theta_0_pred - theta_init) ** 2)
    
    return loss_eq + loss_init

# Initialize model and optimizer
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
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

# Plot loss history
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Physics Loss")
plt.show()

# Predict angular position θ over time
t_test = torch.linspace(t_min, t_max, 300).view(-1, 1)
theta_pred = model(t_test).detach().numpy()

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pendulum Simulation - PINN")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)

# Pendulum origin
origin = (WIDTH // 2, HEIGHT // 3)
length_px = 200  # Pendulum length in pixels

# Clock to control simulation speed
clock = pygame.time.Clock()

# Simulation loop in pygame
running = True
i = 0
while running:
    screen.fill(WHITE)  # Background

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get the current angle from the prediction
    theta = theta_pred[i % len(theta_pred)][0]  # Angle in radians
    i += 1  # Advance in the simulation

    # Calculate pendulum position
    x = origin[0] + length_px * np.sin(theta)
    y = origin[1] + length_px * np.cos(theta)

    # Draw pendulum
    pygame.draw.line(screen, BLACK, origin, (x, y), 4)  # String
    pygame.draw.circle(screen, RED, (int(x), int(y)), 15)  # Pendulum bob

    pygame.display.flip()  # Update screen
    clock.tick(180)  # Control update speed (FPS)

pygame.quit()
sys.exit()
