import csv
import sys
from math import sqrt
from scipy.stats import t

def calc_se(o2, n, x_prom, sxx):
    if n <= 2 or sxx == 0:
        return float('inf'), float('inf')
    se_beta1 = sqrt(o2 / sxx)
    se_beta0 = sqrt(o2 * (1/n + (x_prom**2 / sxx)))
    return se_beta0, se_beta1

def intervalo_confianza(valor, se, n, alpha=0.05):
    if n <= 2:
        return None
    z = calc_z(alpha, n)
    return (valor - z * se, valor + z * se)

def calc_z(alpha, n):
    return t.ppf(1 - alpha / 2, df=n - 2)

def calc_se_respuesta_media(o2, n, x_prom, sxx, x_k):
    if sxx == 0:
        return float('inf')
    return sqrt(o2 * (1/n + ((x_k - x_prom)**2 / sxx)))

def calc_se_prediccion(o2, n, x_prom, sxx, x_k):
    if sxx == 0:
        return float('inf')
    return sqrt(o2 * (1 + 1/n + ((x_k - x_prom)**2 / sxx)))

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

    print('Elija la variable dependiente (target) por su índice:')
    for i, columna in enumerate(cabecera):
        print(f'[{i}]: {columna}')
    
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

    sumatorias_x = {i: 0.0 for i in range(len(cabecera))}
    sumatorias_y = {i: 0.0 for i in range(len(cabecera))}
    sumatorias_x_cuadrado = {i: 0.0 for i in range(len(cabecera))}
    sumatorias_y_cuadrado = {i: 0.0 for i in range(len(cabecera))}
    sumatorias_xy = {i: 0.0 for i in range(len(cabecera))}
    n_validos = {i: 0 for i in range(len(cabecera))}

    for i in range(len(cabecera)):
        if i == target_index:
            continue
        for fila in datos:
            try:
                val_x = float(fila[i])
                val_y = float(fila[target_index])
                
                sumatorias_x[i] += val_x
                sumatorias_y[i] += val_y
                sumatorias_x_cuadrado[i] += val_x**2
                sumatorias_y_cuadrado[i] += val_y**2
                sumatorias_xy[i] += val_x * val_y
                n_validos[i] += 1
            except (ValueError, IndexError):
                pass

    with open(f'resultados_regresion_{cabecera[target_index]}.csv', mode='w', newline='', encoding='utf-8') as archivo_salida:

        escritor_regresion = csv.writer(archivo_salida)
        
        escritor_regresion.writerow(['Variable Independiente', 'Variable Dependiente', 'Beta0', 'Beta1', 'R^2', 'R', 'O^2', 'IC Beta0', 'IC Beta1'])

        for i in range(len(cabecera)):
            if i == target_index or n_validos[i] <= 2:
                continue

            n = n_validos[i]
            x_prom = sumatorias_x[i] / n
            y_prom = sumatorias_y[i] / n
            
            sxx = sumatorias_x_cuadrado[i] - (sumatorias_x[i]**2 / n)
            sxy = sumatorias_xy[i] - (sumatorias_x[i] * sumatorias_y[i] / n)
            
            beta1 = sxy / sxx if sxx != 0 else 0
            beta0 = y_prom - beta1 * x_prom

            sce = 0.0
            stc = 0.0
            for fila in datos:
                try:
                    val_x = float(fila[i])
                    val_y = float(fila[target_index])
                    prediccion = beta0 + beta1 * val_x
                    sce += (val_y - prediccion)**2
                    stc += (val_y - y_prom)**2
                except (ValueError, IndexError):
                    pass
            
            r_cuadrado = 1 - (sce / stc) if stc != 0 else 0
            r = sqrt(max(0, r_cuadrado))
            o_cuadrado = sce / (n - 2)

            se_b0, se_b1 = calc_se(o_cuadrado, n, x_prom, sxx)
            ic_b0 = intervalo_confianza(beta0, se_b0, n)
            ic_b1 = intervalo_confianza(beta1, se_b1, n)
            t_stat = beta1 / se_b1 if se_b1 != 0 else 0

            ic_b0_str = f"[{ic_b0[0]:.4f}, {ic_b0[1]:.4f}]" if ic_b0 else "N/A"
            ic_b1_str = f"[{ic_b1[0]:.4f}, {ic_b1[1]:.4f}]" if ic_b1 else "N/A"

            escritor_regresion.writerow([cabecera[i], cabecera[target_index], beta0, beta1, r_cuadrado, r, o_cuadrado, ic_b0_str, ic_b1_str])

    print("Proceso completado.")
    print("Resultados de la regresión guardados en 'resultados_regresion.csv'")

except FileNotFoundError:
    print(f'Error: El archivo "{nombre_archivo}" no fue encontrado.')
except Exception as e:
    print(f'Ocurrió un error inesperado: {e}')