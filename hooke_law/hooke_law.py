import pygame
import math

WHITE = (255, 255, 255)

class Hooke():

    def __init__(self, width, height):
        self.WIDTH = width
        self.HEIGHT = height
        self.COLOR = (0, 0, 0)

        self.k = 0.1
        self.mass = 5
        self.damping = 0.99
        self.x_equilibrium = self.WIDTH // 2
        self.displacement = 100
        self.velocity = 0

        self.force = self.hooke_force(self.displacement)

        self.acceleration = self.force / self.mass

    def hooke_force(self, x):
        return -self.k * x

def main():
    pygame.init()

    info = pygame.display.Info()
    WIDTH = info.current_h
    HEIGHT = info.current_w

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    
    while True:
        screen.fill(WHITE)

        h = Hooke(WIDTH, HEIGHT)

        h.velocity += h.acceleration
        h.velocity *= h.damping
        h.displacement += h.velocity

        ball_x = int(h.x_equilibrium + h.displacement)
        pygame.draw.line(screen, h.COLOR, (h.x_equilibrium, HEIGHT // 2), (ball_x, HEIGHT // 2), 5)
        pygame.draw.circle(screen, h.COLOR, (ball_x, HEIGHT // 2), 20)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()