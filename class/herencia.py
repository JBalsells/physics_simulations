class Planeta:
    def __init__(self, nombre, radio_km, masa_kg, gravedad_m_s2):
        self.nombre = nombre
        self.radio_km = radio_km
        self.masa_kg = masa_kg
        self.gravedad_m_s2 = gravedad_m_s2

    def volumen(self):
        from math import pi
        return (4/3) * pi * (self.radio_km * 1000) ** 3

    def densidad(self):
        return self.masa_kg / self.volumen()

    def mostrar_info(self):
        print(f" {self.nombre}")
        print(f" Radio: {self.radio_km} km")
        print(f" Masa: {self.masa_kg:.2e} kg")
        print(f" Gravedad: {self.gravedad_m_s2} m/s²")
        print(f" Densidad: {self.densidad():.2f} kg/m³\n")


class PlanetaRocoso(Planeta):
    def __init__(self, nombre, radio_km, masa_kg, gravedad_m_s2, tiene_atmosfera):
        super().__init__(nombre, radio_km, masa_kg, gravedad_m_s2)
        self.tiene_atmosfera = tiene_atmosfera

    def mostrar_info(self):
        super().mostrar_info()
        print(f" Tiene atmósfera: {'Sí' if self.tiene_atmosfera else 'No'}\n")


class PlanetaGaseoso(Planeta):
    def __init__(self, nombre, radio_km, masa_kg, gravedad_m_s2, tipo_gas_principal):
        super().__init__(nombre, radio_km, masa_kg, gravedad_m_s2)
        self.tipo_gas_principal = tipo_gas_principal

    def mostrar_info(self):
        super().mostrar_info()
        print(f" Gas Principal: {self.tipo_gas_principal}\n")


if __name__ == "__main__":
    tierra = PlanetaRocoso("Tierra", radio_km=6371, masa_kg=5.97e24, gravedad_m_s2=9.81, tiene_atmosfera=True)
    marte = PlanetaRocoso("Marte", radio_km=3389, masa_kg=6.39e23, gravedad_m_s2=3.71, tiene_atmosfera=True)
    jupiter = PlanetaGaseoso("Júpiter", radio_km=69911, masa_kg=1.90e27, gravedad_m_s2=24.79, tipo_gas_principal="Hidrógeno")

    tierra.mostrar_info()
    marte.mostrar_info()
    jupiter.mostrar_info()
