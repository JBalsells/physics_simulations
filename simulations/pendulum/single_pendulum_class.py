import pygame
import numpy as np

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Pendulum():
    def __init__(self):
        self.g = 9.81
        self.L = 150
        self.m = 10

        self.theta = np.pi / 4
        self.omega = 0
        self.dt = 0.01

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True

pendulum = Pendulum()

while running:
    screen.fill(WHITE)
    
    alpha = - (pendulum.g / pendulum.L) * np.sin(pendulum.theta)
    
    pendulum.omega += alpha * pendulum.dt
    pendulum.theta += pendulum.omega * pendulum.dt
    
    x = int(WIDTH / 2 + pendulum.L * np.sin(pendulum.theta))
    y = int(HEIGHT / 3 + pendulum.L * np.cos(pendulum.theta))
    
    pygame.draw.line(screen, BLACK, (WIDTH // 2, HEIGHT // 3), (x, y), 2)
    pygame.draw.circle(screen, BLACK, (x, y), 10)
    
    pygame.display.flip()
    clock.tick(480)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
