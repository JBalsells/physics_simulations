import multiprocessing
import random
import time

num_procesos = 20

def tarea(index, nombre):
    print(f"Iniciando {nombre} {index}")
    
    tiempo_aleatorio = random.uniform(2, 3)
    time.sleep(tiempo_aleatorio)
    
    print(f"Finalizando {nombre} {index}")

if __name__ == "__main__":
    procesos = []
    for i in range(num_procesos):
        # time.sleep(0.5)
        proceso = multiprocessing.Process(target=tarea, args=(i,"proceso"))
        procesos.append(proceso)
        proceso.start()

    for proceso in procesos:
        proceso.join()

    print("Procesos Finalizados.")