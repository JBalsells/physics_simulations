import numpy as np
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
    def generate(self, c):
        x = np.linspace(self.xmin, self.xmax, self.width)
        y = np.linspace(self.ymin, self.ymax, self.height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        Z = np.zeros_like(C)
        mandelbrot_set = np.zeros(C.shape, dtype=int)
        
        for i in range(self.max_iter):
            Z = Z ** 2 + C
            mask = np.abs(Z) >= 2
            mandelbrot_set[mask & (mandelbrot_set == 0)] = i
        
        return mandelbrot_set

class Julia(Fractal):
    def __init__(self, xmin, xmax, ymin, ymax, width, height, max_iter, c):
        super().__init__(xmin, xmax, ymin, ymax, width, height, max_iter)
        self.c = c

    def generate(self):
        x = np.linspace(self.xmin, self.xmax, self.width)
        y = np.linspace(self.ymin, self.ymax, self.height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        julia_set = np.zeros(Z.shape, dtype=int)

        for i in range(self.max_iter):
            Z = Z ** 2 + self.c
            mask = np.abs(Z) >= 2
            julia_set[mask & (julia_set == 0)] = i
        
        return julia_set

if __name__ == "__main__":
    c = -0.7 + 0.6j
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width, height = 500, 500
    max_iter = 100

    # Generar conjunto Mandelbrot con un punto fijo c
    mandelbrot = Mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
    mandelbrot_image = mandelbrot.generate(c)

    # Generar conjunto Julia con el mismo valor de c
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
