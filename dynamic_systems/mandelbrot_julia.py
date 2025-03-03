import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Fractal(ABC):    
    def __init__(self, xmin, xmax, ymin, ymax, width, height, max_iter):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.max_iter = max_iter
        
    @abstractmethod
    def generate(self):
        pass

class Mandelbrot(Fractal):    
    def generate(self):
        # Generar una lista de puntos en el eje X y Y de forma manual
        x = [self.xmin + (self.xmax - self.xmin) * i / (self.width - 1) for i in range(self.width)]
        y = [self.ymin + (self.ymax - self.ymin) * i / (self.height - 1) for i in range(self.height)]
        
        mandelbrot_set = [[0] * self.width for _ in range(self.height)]

        # Iterar sobre todos los puntos de la malla
        for ix in range(self.width):
            for iy in range(self.height):
                c = complex(x[ix], y[iy])
                z = 0
                for i in range(self.max_iter):
                    z = z * z + c
                    if abs(z) >= 2:
                        mandelbrot_set[iy][ix] = i
                        break
        
        return mandelbrot_set

class Julia(Fractal):
    def __init__(self, xmin, xmax, ymin, ymax, width, height, max_iter, c):
        super().__init__(xmin, xmax, ymin, ymax, width, height, max_iter)
        self.c = c

    def generate(self):
        # Generar una lista de puntos en el eje X y Y de forma manual
        x = [self.xmin + (self.xmax - self.xmin) * i / (self.width - 1) for i in range(self.width)]
        y = [self.ymin + (self.ymax - self.ymin) * i / (self.height - 1) for i in range(self.height)]
        
        julia_set = [[0] * self.width for _ in range(self.height)]

        # Iterar sobre todos los puntos de la malla
        for ix in range(self.width):
            for iy in range(self.height):
                z = complex(x[ix], y[iy])
                for i in range(self.max_iter):
                    z = z * z + self.c
                    if abs(z) >= 2:
                        julia_set[iy][ix] = i
                        break
        
        return julia_set

if __name__ == "__main__":
    c = 0 + 1j
    xmin, xmax, ymin, ymax = -5.0, 5.0, -2, 2
    width, height = 500, 500
    max_iter = 100
    
    mandelbrot = Mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
    mandelbrot_image = mandelbrot.generate()

    julia = Julia(xmin, xmax, ymin, ymax, width, height, max_iter, c)
    julia_image = julia.generate()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(mandelbrot_image, cmap='hot', extent=(xmin, xmax, ymin, ymax))
    plt.title("Mandelbrot Set with c = {}".format(c))
    plt.subplot(1, 2, 2)
    plt.imshow(julia_image, cmap='hot', extent=(xmin, xmax, ymin, ymax))
    plt.title("Julia Set with c = {}".format(c))
    plt.show()
