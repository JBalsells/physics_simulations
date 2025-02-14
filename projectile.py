import math
import time
import pygame
import random

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Projectile Launcher Simulator")

g = 9.81
dt = 0.005
WHITE = (255, 255, 255)

class Projectile():
    def __init__(self):
        self.initial_angle = random.randint(0,90)
        self.initial_speed = random.randint(1,100)
        self.mass = random.randint(1,10)
        self.rebound_coefficient = random.uniform(0.1, 0.7)
        self.colour = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        self.trajectory = self.projectile_launch()

    def convert_coords(self, x, y):
        return int(x), HEIGHT - int(y)
    
    def rebound_mass_coef(self):
        return max(0.4, self.rebound_coefficient - (self.mass / 50))
    
    def projectile_launch(self):
        rad_angle = math.radians(self.initial_angle)
        vx = self.initial_speed * math.cos(rad_angle)
        vy = self.initial_speed * math.sin(rad_angle)
        
        x, y = 50, 50
        coordinates_trajectory = [(x, y)]
        
        while x < WIDTH:
            x += vx * dt
            vy -= g * dt
            y += vy * dt
            
            if y <= 50:
                y = 50
                rebound_coef = self.rebound_mass_coef()
                vy = -vy * rebound_coef
                
                print(f"Masa: {self.mass} kg, Coef. de rebote: {rebound_coef:.2f}, Nueva velocidad: {vy:.2f} m/s")

                if abs(vy) < 1:  # Condición de parada si la velocidad es muy baja
                    break
            
            coordinates_trajectory.append((x, y))
        
        return coordinates_trajectory
    

def main():
    running = True

    proyectile = Projectile()

    index = 0
    
    while running:
        screen.fill(WHITE)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if index < len(proyectile.trajectory):
            print(proyectile.trajectory[index])
            # pygame.draw.circle(screen, proyectile.colour, Projectile.convert_coords(*proyectile.trajectory[index]), proyectile.mass*2)
            # index += 1
            # time.sleep(dt/10)  # Reducir tiempo de espera para animación más fluida
        
        # pygame.display.flip()
        
    # pygame.quit()

if __name__ == "__main__":
    main()
