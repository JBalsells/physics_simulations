import pygame
import numpy as np

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

g = 9.81
L1, L2 = 100, 100
m1, m2 = 10, 1

theta1, theta2 = np.pi / 2, np.pi / 2
omega1, omega2 = 0, 0

dt = 0.02

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True

while running:
    screen.fill(WHITE)
    
    # Cálculo de aceleraciones angulares
    num1 = -g * (2 * m1 + m2) * np.sin(theta1)
    num2 = -m2 * g * np.sin(theta1 - 2 * theta2)
    num3 = -2 * np.sin(theta1 - theta2) * m2 * (omega2**2 * L2 + omega1**2 * L1 * np.cos(theta1 - theta2))
    den = L1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2))
    alpha1 = (num1 + num2 + num3) / den
    
    num4 = 2 * np.sin(theta1 - theta2) * (omega1**2 * L1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + omega2**2 * L2 * m2 * np.cos(theta1 - theta2))
    den2 = L2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2))
    alpha2 = num4 / den2
    
    # Actualización de velocidades y posiciones
    omega1 += alpha1 * dt
    omega2 += alpha2 * dt
    theta1 += omega1 * dt
    theta2 += omega2 * dt
    
    # Posiciones de los péndulos
    x1 = int(WIDTH / 2 + L1 * np.sin(theta1))
    y1 = int(HEIGHT / 3 + L1 * np.cos(theta1))
    x2 = int(x1 + L2 * np.sin(theta2))
    y2 = int(y1 + L2 * np.cos(theta2))
    
    pygame.draw.line(screen, BLACK, (WIDTH // 2, HEIGHT // 3), (x1, y1), 2)
    pygame.draw.line(screen, BLACK, (x1, y1), (x2, y2), 2)
    pygame.draw.circle(screen, BLACK, (x1, y1), 10)
    pygame.draw.circle(screen, BLACK, (x2, y2), 10)
    
    pygame.display.flip()
    clock.tick(240)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()