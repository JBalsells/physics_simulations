import random
import math

class FiguraGeometrica:
    def __init__(self, nombre):
        self.nombre = nombre

    def generar_punto(self):
        raise NotImplementedError("Este método debe ser implementado en las subclases")

class Cuadrado(FiguraGeometrica):
    def __init__(self, lado):
        super().__init__("Cuadrado")
        self.lado = lado

    def generar_punto(self):
        x = random.uniform(-self.lado / 2, self.lado / 2)
        y = random.uniform(-self.lado / 2, self.lado / 2)
        return x, y

class Circulo(FiguraGeometrica):
    def __init__(self, radio):
        super().__init__("Círculo")
        self.radio = radio

    def esta_dentro(self, x, y):
        return x**2 + y**2 <= self.radio**2

class MonteCarloPi:
    def __init__(self, num_puntos, cuadrado, circulo):
        self.num_puntos = num_puntos
        self.cuadrado = cuadrado
        self.circulo = circulo
        self.dentro_circulo = 0

    def calcular_pi(self):
        for _ in range(self.num_puntos):
            x, y = self.cuadrado.generar_punto()
            if self.circulo.esta_dentro(x, y):
                self.dentro_circulo += 1
        return 4 * (self.dentro_circulo / self.num_puntos)

if __name__ == "__main__":
    lado_cuadrado = 2  # El cuadrado de referencia es de lado 2 (-1 a 1 en cada eje)
    radio_circulo = 1   # El círculo inscrito tiene radio 1
    num_puntos = 10000
    
    cuadrado = Cuadrado(lado_cuadrado)
    circulo = Circulo(radio_circulo)
    
    estimador_pi = MonteCarloPi(num_puntos, cuadrado, circulo)
    print(f"Estimación de π: {estimador_pi.calcular_pi()}")
