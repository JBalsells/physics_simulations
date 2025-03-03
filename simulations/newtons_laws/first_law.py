import pygame

# Inicializar pygame
pygame.init()

# Configuración de la pantalla
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Primera Ley de Newton - Inercia con Fricción")

# Colores
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Propiedades del objeto
x, y = 100, HEIGHT // 2
vx = 3  # Velocidad inicial
friccion = 0.01  # Coeficiente de fricción

running = True

while running:
    screen.fill(WHITE)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Aplicar fricción si el objeto aún se mueve
    if abs(vx) > 0:
        vx -= friccion * (1 if vx > 0 else -1)  # Reducir velocidad en dirección opuesta
    
    # Evitar que la velocidad se vuelva negativa y haga que el objeto retroceda
    if abs(vx) < 0.01:
        vx = 0  # Detener el objeto completamente
    
    x += vx  # Actualizar posición
    
    pygame.draw.circle(screen, RED, (int(x), int(y)), 20)
    
    pygame.display.flip()
    pygame.time.delay(10)

pygame.quit()
