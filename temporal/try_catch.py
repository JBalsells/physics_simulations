def division_segura():
    try:
        num1 = float(input("Introduce el primer número: "))
        num2 = float(input("Introduce el segundo número: "))
        resultado = num1 / num2
    except ZeroDivisionError:
        return("Error: No se puede dividir entre cero.")
    except ValueError:
        return("Error: Debes ingresar números válidos.")
    else:
        return(f"El resultado es: {resultado}")

def acceder_lista():
    try:
        numeros = [10, 20, 30, 40, 50]
        print("Lista:", numeros)
        indice = int(input("Introduce un índice para acceder al elemento: "))
        return(f"El elemento en la posición {indice} es {numeros[indice]}")
    except ValueError:
        return("Error: Debes ingresar un número entero válido.")
    except IndexError:
        return("Error: Índice fuera de rango.")

if __name__ == "__main__":
    #response = division_segura()
    response = acceder_lista()