"""
Author: Jorge A. Balsells Orellana
Date: March 6 2025
Description: Physics-Informed Neural Network (PINN) for physics simulatinons.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pygame

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2)  # Output: [x, y]
        )
    
    def forward(self, t):
        return self.net(t)

# Problem parameters
m = 1.0  # kg (mass of the projectile)
k = 0.1  # Air resistance coefficient
g = 9.81  # m/s^2 (gravity)
v0 = 20.0  # m/s (initial velocity)
angle = np.radians(45)  # Initial angle
vx0 = v0 * np.cos(angle)
vy0 = v0 * np.sin(angle)
restitution = 0.7  # Coefficient of restitution (bouncing factor)

# Create and train the PINN model
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training data
t_data = torch.linspace(0, 5, 200).view(-1, 1)  # Extended time range
x_exact = vx0 * t_data - (k/m) * t_data**2
y_exact = vy0 * t_data - 0.5 * g * t_data**2 - (k/m) * t_data**2
targets = torch.cat([x_exact, y_exact], dim=1)

# Training loop
for epoch in range(5000):
    optimizer.zero_grad()
    output = model(t_data)
    loss = loss_fn(output, targets)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Pygame for visualization
pygame.init()
screen = pygame.display.set_mode((800, 400))
clock = pygame.time.Clock()

# Scale coordinates for Pygame
offset_x, offset_y = 50, 300
scale = 20

# Real-time simulation variables
x, y = 0, 0
vx, vy = vx0, vy0
dt = 0.05  # Time step in seconds
bounces = 0
running = True
trajectory = []

while running:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Update physics
    x += vx * dt
    vy -= g * dt
    y += vy * dt
    
    # Bounce condition
    if y <= 0:
        y = 0
        vy = -vy * restitution
        vx *= restitution  # Reduce horizontal speed due to friction
        bounces += 1
        if abs(vy) < 0.5:  # Stop if velocity is too low
            break
    
    # Store trajectory
    trajectory.append((x, y))
    
    # Draw trajectory
    for px, py in trajectory:
        pygame.draw.circle(screen, (255, 0, 0), (int(offset_x + px * scale), int(offset_y - py * scale)), 3)
    
    # Draw projectile
    pygame.draw.circle(screen, (0, 255, 0), (int(offset_x + x * scale), int(offset_y - y * scale)), 5)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
