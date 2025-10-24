import csv
import sys
import math

def estandarizar_datos(datos, independientes_indices):

    datos_escalados = [fila[:] for fila in datos] 
    parametros_escala = {} 

    for idx in independientes_indices:
        columna = []
        for fila in datos:
            try:
                columna.append(float(fila[idx]))
            except ValueError:
                continue 

        if not columna:
            continue

        n = len(columna)
        media = sum(columna) / n
        desviacion_std = math.sqrt(sum([(val - media)**2 for val in columna]) / n)
        
        # Se guarda la media y desviación estándar para revertir la estandarizacion
        parametros_escala[idx] = (media, desviacion_std)

        for i in range(len(datos_escalados)):
            if desviacion_std > 0:
                val_original = float(datos_escalados[i][idx])
                datos_escalados[i][idx] = (val_original - media) / desviacion_std
            else:
                # Si la desviación es 0, todos los valores son iguales
                datos_escalados[i][idx] = 0.0
                
    return datos_escalados, parametros_escala

def descenso_gradiente(datos, target_index, independientes_indices, coord_origen, pendientes):
    learning_rate = 0.005
    num_iteraciones = 2500
    m = float(len(datos)) 

    if m == 0:
        print("No hay datos para entrenar.")
        return

    for i in range(num_iteraciones):
        grad_b0 = 0.0  
        grad_pendientes = [0.0 for _ in independientes_indices] 

        for fila in datos:
            y = float(fila[target_index]) 
            
            prediccion = coord_origen
            for j, index in enumerate(independientes_indices):
                prediccion += pendientes[j] * float(fila[index]) 
            
            error = prediccion - y
            
            grad_b0 += error
            for j, index in enumerate(independientes_indices):
                x_j_escalado = float(fila[index])
                grad_pendientes[j] += error * x_j_escalado

        grad_b0 /= m
        for j in range(len(grad_pendientes)):
            grad_pendientes[j] /= m

        coord_origen -= learning_rate * grad_b0
        for j in range(len(pendientes)):
            pendientes[j] -= learning_rate * grad_pendientes[j]

        if i % 100 == 0:
            print(f"Iteración {i}: Costo (B0={coord_origen:.4f})")
             
    return coord_origen, pendientes


if len(sys.argv) < 2:
    print("Uso: python regression.py <nombre_archivo.csv>")
    sys.exit(1)
nombre_archivo = sys.argv[1]

try:
    datos = []
    cabecera = []
    with open(nombre_archivo, mode='r', newline='', encoding='utf-8') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        cabecera = next(lector_csv)   
        datos = [fila for fila in lector_csv]
    
    print('Variables disponibles:')
    for i, nombre in enumerate(cabecera):
        print(f'{i}: {nombre}')

    input_index = input('Índice de la variable dependiente: ')
    try:
        target_index = int(input_index)
        if not (0 <= target_index < len(cabecera)):
            raise ValueError
    except (ValueError, TypeError):
        print('Índice inválido. Por favor, ingrese un número válido.')
        exit(1)

    print(f'Has seleccionado la variable dependiente: {cabecera[target_index]}')
    print('-' * 20)

    input_independientes = input('Ingrese los índices de las variables independientes separados por comas (o "all" para todas excepto la dependiente): ')
    if input_independientes.strip().lower() == 'all':
        independientes_indices = [i for i in range(len(cabecera)) if i != target_index]
    else:
        try:
            independientes_indices = [int(idx.strip()) for idx in input_independientes.split(',')]
            print(independientes_indices)
            for idx in independientes_indices:
                if not (0 <= idx < len(cabecera)) or idx == target_index:
                    raise ValueError
        except (ValueError, TypeError):
            print('Índices inválidos. Por favor, ingrese números válidos separados por comas.')
            exit(1)
    print(f'Has seleccionado las variables independientes: {[cabecera[i] for i in independientes_indices]}')

    # Estandarizar los datos
    datos_escalados, parametros = estandarizar_datos(datos, independientes_indices)
   
    coord_origen = 0
    pendientes = [0.0 for _ in independientes_indices]

    # Ejecutar el descenso de gradiente
    coord_origen, pendientes = descenso_gradiente(datos_escalados, target_index, independientes_indices, coord_origen, pendientes)

    # Revertir la estandarización de los coeficientes
    temp = 0
    for i, idx in enumerate(independientes_indices):
        media, std = parametros[idx]
        if std > 0:
            pendientes[i] /= std
        else:
            pendientes[i] = 0.0
        temp += media * pendientes[i]
    coord_origen -= temp

    ecuacion = f"{cabecera[target_index]} = {coord_origen:.4f} "
    for i, idx in enumerate(independientes_indices):
        signo = '+' if pendientes[i] >= 0 else '-'
        ecuacion += f"{signo} {abs(pendientes[i]):.4f}*{cabecera[idx]} "
    print(f"Ecuación de la regresión: {ecuacion.strip()}")


except FileNotFoundError:
    print(f'Error: El archivo "{nombre_archivo}" no fue encontrado.')
except Exception as e:
    print(f'Ocurrió un error inesperado: {e}')