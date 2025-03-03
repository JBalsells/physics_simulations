import pygame

# Inicializar pygame
pygame.init()

# Configuración de la pantalla
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Segunda Ley de Newton - F = m*a con Fricción")

# Colores
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Propiedades del objeto
x, y = 100, HEIGHT // 2
vx = 0  # Velocidad inicial
masa = 10  # Masa del objeto (kg)
fuerza = 50  # Fuerza aplicada (N)
coef_friccion = 0.2  # Coeficiente de fricción (ajustable)
dt = 0.1  # Paso de tiempo

running = True

while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            aceleracion = fuerza / masa  # Segunda Ley de Newton
            vx += aceleracion * dt  # Aplicar la fuerza al presionar ESPACIO
    
    # Aplicar fricción si el objeto está en movimiento
    if abs(vx) > 0:
        fuerza_friccion = coef_friccion * masa * 9.81  # Fuerza de fricción = μ * m * g
        friccion = (fuerza_friccion / masa) * dt  # Aceleración por fricción
        
        # Reducir la velocidad en la dirección opuesta al movimiento
        if abs(vx) > friccion:
            vx -= friccion * (1 if vx > 0 else -1)
        else:
            vx = 0  # Evitar que oscile en valores cercanos a cero

    x += vx  # Actualizar posición

    pygame.draw.circle(screen, RED, (int(x), int(y)), 20)

    pygame.display.flip()
    pygame.time.delay(10)

pygame.quit()
