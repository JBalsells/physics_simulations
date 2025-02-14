import pygame

# Inicializar pygame
pygame.init()

# Configuración de la pantalla
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tercera Ley de Newton - Acción y Reacción")

# Colores
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Propiedades de las esferas
x1, y1 = 300, HEIGHT // 2
x2, y2 = 500, HEIGHT // 2
vx1 = 2  # Velocidad de la primera esfera
vx2 = -2  # Velocidad de la segunda esfera

running = True

while running:
    screen.fill(WHITE)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Movimiento de las esferas
    x1 += vx1
    x2 += vx2
    
    # Detección de colisión
    if abs(x1 - x2) < 40:
        vx1, vx2 = -vx1, -vx2  # Intercambian velocidades (acción y reacción)
    
    pygame.draw.circle(screen, RED, (int(x1), int(y1)), 20)
    pygame.draw.circle(screen, BLUE, (int(x2), int(y2)), 20)
    
    pygame.display.flip()
    pygame.time.delay(10)

pygame.quit()