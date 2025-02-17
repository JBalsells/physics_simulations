import pygame
import math


WIDTH, HEIGHT = 800, 600
G = 1
DT = 0.8

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Body:
    def __init__(self, x, y, mass, vx, vy):
        self.x = x
        self.y = y
        self.mass = mass
        self.vx = vx
        self.vy = vy
        self.radius = int(math.sqrt(mass) * 2)

    def update_position(self):
        self.x += self.vx * DT
        self.y += self.vy * DT

    def apply_gravity(self, other):
        dx = other.x - self.x
        dy = other.y - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2) + 0.1
        force = G * self.mass * other.mass / distance ** 2
        
        acceleration = force / self.mass
        self.vx += acceleration * (dx / distance) * DT
        self.vy += acceleration * (dy / distance) * DT

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación 3 Cuerpos")
clock = pygame.time.Clock()

cx, cy = WIDTH // 2, HEIGHT // 2
r = 100
m = 30
v = 1.5

bodies = [
    Body(cx + r, cy, m, 0, v),
    Body(cx - r / 2, cy + math.sqrt(3) * r / 2, m, -v * math.cos(math.pi / 3), -v * math.sin(math.pi / 3)),
    Body(cx - r / 2, cy - math.sqrt(3) * r / 2, m, v * math.cos(math.pi / 3), -v * math.sin(math.pi / 3))
]

running = True
while running:
    screen.fill(BLACK)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    for i in range(len(bodies)):
        for j in range(len(bodies)):
            if i != j:
                bodies[i].apply_gravity(bodies[j])

    for body in bodies:
        body.update_position()
        pygame.draw.circle(screen, WHITE, (int(body.x), int(body.y)), body.radius)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
