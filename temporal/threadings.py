import threading
import random
import time

num_hilos = 20
# barrier = threading.Barrier(num_hilos)

def tarea(index, nombre):
    print(f"Iniciando {nombre} {index}")
    
    tiempo_aleatorio = random.uniform(2, 3)
    time.sleep(tiempo_aleatorio)
    
    # barrier.wait() 
    print(f"Finalizando {nombre} {index}")

hilos = []
for i in range(num_hilos):

    # time.sleep(0.5)
    
    hilo = threading.Thread(target=tarea, args=(i,"hilo"))
    hilos.append(hilo)
    hilo.start()

for hilo in hilos:
    hilo.join()

print("Hilos Finalizados.")
