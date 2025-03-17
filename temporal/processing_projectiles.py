import math
import pygame
import random
import sys
import multiprocessing

sys.setrecursionlimit(50000)

WHITE = (255, 255, 255)
WIDTH, HEIGHT = 3200, 800
x_reference = WIDTH / 2
y_reference = 50
num_projectiles = 10000

class Projectile:
    def __init__(self, width, height, x_reference, y_reference):
        self.width = width
        self.height = height
        self.dt = 0.1
        self.gravity = 9.81

        # Propiedades aleatorias del proyectil
        self.mass = random.gauss(0.7, 0.2)
        self.initial_angle = random.uniform(-90, 90)
        self.initial_speed = random.gauss(50, 10)
        self.friction = random.gauss(0.8, 0.1)
        self.colour = (0, 0, 0)

        self.trajectory = self.__recursive_projectile_launch(x_reference, y_reference)

    def __recursive_projectile_launch(self, x, y, vx=None, vy=None, trajectory=None):
        if trajectory is None:
            trajectory = [(x, y)]
            rad_angle = math.radians(abs(self.initial_angle))
            vx = self.initial_speed * math.cos(rad_angle)
            vy = self.initial_speed * math.sin(rad_angle)

            if self.initial_angle < 0:
                vx = -vx  

        if not (0 < x < self.width):
            return trajectory

        x += vx * self.dt
        vy -= self.gravity * self.dt
        y += vy * self.dt

        if y <= 50:
            y = 50
            friction_effect = 1 - (self.friction * (self.mass / 10))
            vy = -vy * max(0, friction_effect)

            print(f"Masa: {self.mass:.2f} kg, Ángulo: {self.initial_angle}°, Fricción: {self.friction:.2f}, Nueva velocidad: {vy:.2f} m/s")

            if abs(vy) < 1:
                return trajectory

        trajectory.append((x, y))

        return self.__recursive_projectile_launch(x, y, vx, vy, trajectory)

def convert_coords(x, y):
    return int(x), HEIGHT - int(y)

def create_projectile(index):
    """Función para instanciar un proyectil en un proceso separado."""
    projectile = Projectile(WIDTH, HEIGHT, x_reference, y_reference)
    return projectile

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Projectile Launcher Simulator")

    # Usar Pool para manejar múltiples procesos
    with multiprocessing.Pool() as pool:
        projectiles = pool.map(create_projectile, range(num_projectiles))

    # Iniciar simulación en Pygame
    running = True
    index = 0
    max_length = max(len(p.trajectory) for p in projectiles)

    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for p in projectiles:
            if index < len(p.trajectory):
                pygame.draw.circle(screen, p.colour, convert_coords(*p.trajectory[index]), int(p.mass * 3))

        index += 1
        if index >= max_length:
            break

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
