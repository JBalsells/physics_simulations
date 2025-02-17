import time
import math
import pygame
import random

WHITE = (255, 255, 255)
WIDTH, HEIGHT = 3200, 800
x_reference = WIDTH/2
y_reference = 50

class Projectile():
    def __init__(self, width, height, x_reference, y_reference):
        
        mu_speed = 50
        sigma_speed = 20

        mu_mass = 0.7
        sigma_mass = 0.5

        self.width = width
        self.height = height
        self.dt = 0.005
        self.gravity = 9.81
        self.mass = random.gauss(mu_mass, sigma_mass)
        self.initial_angle = random.uniform(-90, 90)
        self.initial_speed = random.gauss(mu_speed, sigma_speed)
        self.rebound_coefficient = random.gauss(0.5, 0.4)
        self.colour = (255,150,50)

        self.trajectory = self.__projectile_launch(x_reference, y_reference)

    def __rebound_mass_coef(self):
        return max(0.4, self.rebound_coefficient - (self.mass / 50))

    def __projectile_launch(self, x, y):
        rad_angle = math.radians(abs(self.initial_angle))
        vx = self.initial_speed * math.cos(rad_angle)
        vy = self.initial_speed * math.sin(rad_angle)

        if self.initial_angle < 0:
            vx = -vx  

        coordinates_trajectory = [(x, y)]

        while 0 < x < self.width:
            x += vx * self.dt
            vy -= self.gravity * self.dt
            y += vy * self.dt

            if y <= 50:
                y = 50
                rebound_coef = self.__rebound_mass_coef()
                vy = -vy * rebound_coef

                print(f"Masa: {self.mass:.2f} kg, Ángulo: {self.initial_angle}°, Coef. de rebote: {rebound_coef:.2f}, Nueva velocidad: {vy:.2f} m/s")

                if abs(vy) < 1:
                    break

            coordinates_trajectory.append((x, y))

        return coordinates_trajectory

def convert_coords(x, y):
    return int(x), HEIGHT - int(y)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Projectile Launcher Simulator")

    running = True
    projectiles = [Projectile(WIDTH, HEIGHT, x_reference, y_reference) for _ in range(50)]

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

        # time.sleep(0.0001)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
