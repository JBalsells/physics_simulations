import pygame
import numpy as np

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

g = 9.81
L = 150
m = 10

theta = np.pi / 4  # Ángulo inicial
omega = 0  # Velocidad angular inicial

dt = 0.01  # Paso de tiempo

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True

while running:
    screen.fill(WHITE)
    
    alpha = - (g / L) * np.sin(theta)
    
    omega += alpha * dt
    theta += omega * dt
    
    x = int(WIDTH / 2 + L * np.sin(theta))
    y = int(HEIGHT / 3 + L * np.cos(theta))
    
    pygame.draw.line(screen, BLACK, (WIDTH // 2, HEIGHT // 3), (x, y), 2)
    pygame.draw.circle(screen, BLACK, (x, y), 10)
    
    pygame.display.flip()
    clock.tick(480)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
