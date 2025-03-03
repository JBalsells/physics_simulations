class Particle:
    def __init__(self, masa, velocidad):
        self.masa = masa
        self.velocidad = velocidad

    def energia_cinetica(self):
        return (1/2) * self.masa * self.velocidad ** 2

class CuerpoRigido:
    def __init__(self, momento_inercia, angular_velocidad):
        self.momento_inercia = momento_inercia
        self.angular_velocidad = angular_velocidad

    def energia_rotacional(self):
        return (1/2) * self.momento_inercia * self.angular_velocidad ** 2

class ParticulaEnMovimiento(Particle, CuerpoRigido):
    def __init__(self, masa, velocidad, momento_inercia, angular_velocidad):
        Particle.__init__(self, masa, velocidad)
        CuerpoRigido.__init__(self, momento_inercia, angular_velocidad)

    def energia_total(self):
        return self.energia_cinetica() + self.energia_rotacional()

particula = ParticulaEnMovimiento(masa=2.0, velocidad=3.0, momento_inercia=5.0, angular_velocidad=4.0)

print("Energía cinética:", particula.energia_cinetica())
print("Energía rotacional:", particula.energia_rotacional())
print("Energía total:", particula.energia_total())
