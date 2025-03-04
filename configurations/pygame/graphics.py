import pygame

class Graphics():
    def __init__(self):
        pygame.init()
        
        self.info = pygame.display.Info()
        self.WIDTH = self.info.current_w
        self.HEIGHT = self.info.current_h

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT)) 
        
    def set_caption(self, caption):
        pygame.display.set_caption(caption)
        
    def close(self):
        pygame.quit()
       

def graphics_cycle(func):
    def wrapper():
        running = True
        while running:
            graphics.screen.fill((255,255,255))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            func()

            pygame.display.flip()
    return wrapper
 
if __name__ == "__main__":
    
    @graphics_cycle
    def saludar():
        print("¡Hola!")
        
    graphics = Graphics()
    saludar()
    graphics.close()
