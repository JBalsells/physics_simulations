import pygame
import math

class Hooke():
    def __init__(self, width):
        self.k = 0.1
        self.mass = 5
        self.damping = 0.99
        self.x_equilibrium = width // 2
        self.displacement = 100
        self.velocity = 0
    def hooke_force(self, x):
        return -self.k * x
    def hooke_calculus(self):
        force = self.hooke_force(self.displacement)
        acceleration = force / self.mass
        self.velocity += acceleration
        self.velocity *= self.damping
        self.displacement += self.velocity

def main():
    WIDTH, HEIGHT = 800, 400
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    hooke = Hooke(WIDTH)

    while running:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        hooke.hooke_calculus()
        
        ball_x = int(hooke.x_equilibrium + hooke.displacement)
        pygame.draw.line(screen, BLACK, (hooke.x_equilibrium, HEIGHT // 2), (ball_x, HEIGHT // 2), 5)
        pygame.draw.circle(screen, BLACK, (ball_x, HEIGHT // 2), 20)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()
