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
masa = 5  # Masa del proyectil en kg
coef_rebote_base = 0.7  # Coeficiente de restitución base

# Convertir coordenadas a las de pygame
def convertir_coordenadas(x, y):
    return int(x), HEIGHT - int(y)

def coef_rebote_masa(m):
    """Calcula un coeficiente de restitución dependiente de la masa"""
    return max(0.4, coef_rebote_base - (m / 50))  # Evita coeficientes negativos

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
            coef_rebote = coef_rebote_masa(masa)  # Obtener coef. según masa
            vy = -vy * coef_rebote
            
            print(f"Masa: {masa} kg, Coef. de rebote: {coef_rebote:.2f}, Nueva velocidad: {vy:.2f} m/s")

            if abs(vy) < 1:  # Condición de parada si la velocidad es muy baja
                break
        
        trayectoria.append((x, y))
    
    return trayectoria

def main():
    running = True
    trayectoria = lanzar_proyectil(70, 65)  # Velocidad de 70 m/s, ángulo de 80°
    
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
