import pygame

WIDTH, HEIGHT = 400, 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

g = 9.81  # Gravedad (m/s^2)
k = 0.5  # Constante del resorte
mass = 5  # Masa del objeto
damping = 0.95  # Amortiguamiento
x_equilibrium = HEIGHT // 4  # Posición de equilibrio inicial
displacement = 150  # Desplazamiento inicial
velocity = 0  # Velocidad inicial

def hooke_force(x):
    return -k * x  # Ley de Hooke

def gravity_force():
    return mass * g  # Fuerza gravitacional

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
                
        # Aplicar fuerzas
        force = hooke_force(displacement) + gravity_force()
        acceleration = force / mass
        velocity += acceleration
        velocity *= damping  # Aplicar amortiguamiento
        displacement += velocity
        
        # Dibujar el resorte
        ball_y = int(x_equilibrium + displacement)
        pygame.draw.line(screen, BLACK, (WIDTH // 2, x_equilibrium), (WIDTH // 2, ball_y), 5)
        pygame.draw.circle(screen, BLACK, (WIDTH // 2, ball_y), 20)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()
