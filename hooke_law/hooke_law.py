import pygame
import math

# Configuración inicial
WIDTH, HEIGHT = 800, 400
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Propiedades del resorte
k = 0.1  # Constante del resorte
mass = 5  # Masa del objeto
damping = 0.99  # Amortiguamiento
x_equilibrium = WIDTH // 2  # Posición de equilibrio
displacement = 100  # Desplazamiento inicial
velocity = 0  # Velocidad inicial

def hooke_force(x):
    return -k * x  # Ley de Hooke

def main():
    global displacement, velocity
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    while running:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Aplicar la Ley de Hooke y actualizar la física
        force = hooke_force(displacement)
        acceleration = force / mass
        velocity += acceleration
        velocity *= damping  # Aplicar amortiguamiento
        displacement += velocity
        
        # Dibujar el resorte
        ball_x = int(x_equilibrium + displacement)
        pygame.draw.line(screen, BLACK, (x_equilibrium, HEIGHT // 2), (ball_x, HEIGHT // 2), 5)
        pygame.draw.circle(screen, BLACK, (ball_x, HEIGHT // 2), 20)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()