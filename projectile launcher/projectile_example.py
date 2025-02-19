import math
import pygame

WHITE = (255, 255, 255)
WIDTH, HEIGHT = 1900, 1000
x_reference = 50
y_reference = 50

class Projectile():
    def __init__(self, width, height, x_reference, y_reference):
        self.WIDTH = width
        self.HEIGHT = height
        self.DT = 0.01
        self.GRAVITY = 9.81

        self.mass = 5
        self.initial_angle = 60
        self.initial_speed = 80
        self.friction = 0.5

        self.colour = (0, 0, 0)
        self.radius = int(self.mass*3)

        self.trajectory = self.__projectile_launch(x_reference, y_reference)

    def __projectile_launch(self, x, y):
        rad_angle = math.radians(self.initial_angle)
        v0x = self.initial_speed * math.cos(rad_angle)
        v0y = self.initial_speed * math.sin(rad_angle)

        t_f = (2 * v0y) / self.GRAVITY if y == 0 else (v0y + math.sqrt(v0y**2 + 2 * self.GRAVITY * y)) / self.GRAVITY

        coordinates_trajectory = []
        t = 0

        while t <= t_f and 0 < x < self.WIDTH:
            x_pos = x + v0x * t
            y_pos = y + v0y * t - 0.5 * self.GRAVITY * t**2

            if y_pos <= 0:
                y_pos = 0
                break

            coordinates_trajectory.append((x_pos, y_pos))
            t += self.DT

        return coordinates_trajectory

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Projectile Launcher Simulator")

    running = True
    p = Projectile(WIDTH, HEIGHT, x_reference, y_reference)
    print(p.trajectory)
    """
    index = 0
    max_length = len(p.trajectory)
    while running:
        screen.fill(WHITE)

        if index < len(p.trajectory):
            pygame.draw.circle(screen, p.colour, (p.trajectory[index][0], p.trajectory[index][1]), p.radius)

        index += 1
        if index >= max_length:
            break

        pygame.display.flip()

    pygame.quit()
    """

if __name__ == "__main__":
    main()