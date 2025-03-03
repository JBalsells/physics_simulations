import matplotlib.pyplot as plt

class Movimiento:
    def __init__(self, posicion_inicial, velocidad_inicial, tiempo_total):
        self.posicion_inicial = posicion_inicial
        self.velocidad_inicial = velocidad_inicial
        self.tiempo_total = tiempo_total

    def calcular_posicion(self, t):
        pass

class MRU(Movimiento):
    def calcular_posicion(self, t):
        # Ecuación: x = x0 + v * t
        return self.posicion_inicial + self.velocidad_inicial * t

class MRUA(Movimiento):
    def __init__(self, posicion_inicial, velocidad_inicial, tiempo_total, aceleracion):
        super().__init__(posicion_inicial, velocidad_inicial, tiempo_total)
        self.aceleracion = aceleracion

    def calcular_posicion(self, t):
        # Ecuación: x = x0 + v0*t + (1/2) * a * t^2
        return self.posicion_inicial + self.velocidad_inicial * t + 0.5 * self.aceleracion * t**2

if __name__ == "__main__":
    posicion_inicial = 0
    velocidad_inicial = 5
    tiempo_total = 10
    aceleracion = 2

    movimiento_mru = MRU(posicion_inicial, velocidad_inicial, tiempo_total)
    movimiento_mrua = MRUA(posicion_inicial, velocidad_inicial, tiempo_total, aceleracion)

    t_values = [t for t in range(0, 50 + 1)]
    pos_mru = [movimiento_mru.calcular_posicion(t) for t in t_values]
    pos_mrua = [movimiento_mrua.calcular_posicion(t) for t in t_values]

    plt.figure(figsize=(8, 5))
    plt.plot(t_values, pos_mru, label="MRU (v=5 m/s)", linestyle="--", color="blue")
    plt.plot(t_values, pos_mrua, label="MRUA (v=5 m/s, a=2 m/s²)", linestyle="-", color="red")

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Posición (m)")
    plt.title("Movimiento Rectilíneo Uniforme vs. Uniformemente Acelerado")
    plt.legend()
    plt.grid()
    plt.show()
