def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
    
def factorial_no_recursivo(n):
    resultado = 1
    for i in range(2, n + 1):
        resultado *= i
    return resultado

def fibonacci(n):
    return n if n <= 1 else fibonacci(n - 1) + fibonacci(n - 2)
    
def fibonacci_no_recursivo(n):
    if n == 0:
        return 0
    
    primer_numero = 0
    segundo_numero = 1

    for _ in range(2, n + 1):
        siguiente_numero = primer_numero + segundo_numero
        primer_numero = segundo_numero
        segundo_numero = siguiente_numero
    
    return segundo_numero

def suma_lista(lista):
    if len(lista) == 0:
        return 0
    else:
        return lista[0] + suma_lista(lista[1:])

if __name__ == "__main__":
    #num = int(input("Introduce un número: "))
    #print(f"El factorial calculado en una funcion recursiva de {num} es {factorial(num)}")
    #print(f"El factorial calculado en una funcion NO recursiva de {num} es {factorial_no_recursivo(num)}")
    
    num = int(input("Introduce el término de Fibonacci que deseas calcular: "))
    print(f"El término {num} de Fibonacci en una funcion recursiva es {fibonacci(num)}")
    print(f"El término {num} de Fibonacci en una funcion NO recursiva es {fibonacci_no_recursivo(num)}")
    
    #numeros = [1, 2, 3, 4, 5]
    #print(f"La suma de la lista {numeros} es {suma_lista(numeros)}")