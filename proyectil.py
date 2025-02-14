import pygame
import math
import time

# Inicializar pygame
pygame.init()

# Configuración de la pantalla
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación de Lanzamiento de Proyectil")

# Colores
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Parámetros físicos
g = 9.81  # Gravedad (m/s^2)
dt = 0.005  # Paso de tiempo
coef_rebote = 0.7  # Coeficiente de restitución para el rebote

# Convertir coordenadas a las de pygame
def convertir_coordenadas(x, y):
    return int(x), HEIGHT - int(y)

def lanzar_proyectil(velocidad, angulo):
    angulo_rad = math.radians(angulo)
    vx = velocidad * math.cos(angulo_rad)
    vy = velocidad * math.sin(angulo_rad)
    
    x, y = 50, 50  # Posición inicial
    trayectoria = [(x, y)]
    
    while x < WIDTH:
        x += vx * dt
        vy -= g * dt
        y += vy * dt
        
        if y <= 50:
            y = 50
            vy = -vy * coef_rebote
            if abs(vy) < 1:  # Condición de parada si la velocidad es muy baja
                break
        
        trayectoria.append((x, y))
    
    return trayectoria

def main():
    running = True
    trayectoria = lanzar_proyectil(70, 80)  # Velocidad de 50 m/s, ángulo de 45°
    
    index = 0
    
    while running:
        screen.fill(WHITE)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if index < len(trayectoria):
            pygame.draw.circle(screen, RED, convertir_coordenadas(*trayectoria[index]), 10)
            index += 1
            time.sleep(dt/10)  # Reducir tiempo de espera para animación más fluida
        
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__":
    main()
