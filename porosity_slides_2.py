# Programa que calcula la porosidad slide por slide de una tomografi­a 3D

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time 

from numba import njit
from matplotlib import colors

###############################################################################
# Registramos el tiempo de inicio del script
start_time = time.time()
###############################################################################
# Creamos una nueva carpera donde guardaremos los datos obtenidos
nueva_carpeta = 'HJ1468_P'
###############################################################################
# Directorio dende quieres crear la nueva carpeta 
directorio_destino = r'C:\Users\eduar\Desktop\Ejercicios_Python\porosidades_2D'
# Unimos la nueva carpeta con el directorio destino
ruta_nueva_carpeta = os.path.join(directorio_destino, nueva_carpeta)
# Comprobamos si la carpeta ya existe antes de crearla
if not os.path.exists(ruta_nueva_carpeta):
    os.makedirs(ruta_nueva_carpeta)
###############################################################################
# Ruta de la direccion de la carpeta que contiene todas las imagenes
# input_images_path = r"C:\Users\eduar\Desktop\Recorta_imagenes\0 Prismas Porosidad\HJ1468-P_porosidad_total"
input_images_path = r"C:\Users\eduar\Desktop\Recorta_imagenes\Prismas_binarizados\HJ1468-P_porosidad_total"
###############################################################################
# Guardamos el nombre de cada imagen en una lista llamada: file_names
file_names = os.listdir(input_images_path)

# Creamos un tensor donde guardaremos los valores de los pixeles de cada imagen
tensor = []

# Leemos cada uno de los elementos de la lista 
for x in file_names:
    
    # Construimos el path donde sen encuentran c/u de las imagenes
    image_path = input_images_path + "/" + x
    
    # Leemos y convertimos cada imagen a escala de grises
    image = cv2.imread(image_path, 0)
    # image = cv2.imread(image_path)
    
    # Obtenemos el valor del umbral y binarizamos con el metodo de Otsu
    # umbral, img_bin = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    # tensor.append(img_bin)
    tensor.append(image)
    
# Generamos el tensor tridimensional
matrix = np.asarray(tensor)
print('dimensiones del tensor =', np.shape(matrix))
###############################################################################

max_pixeles = int(len(matrix[0])) # maximo de pixeles que deseamos que tengan las submuestras

# Resoluciones de las imagenes: x micras/pixel
micra = 0.0001 # cm
cm = 10000.00 # micras

res_g = 28.0*micra
res_m = 15.0*micra
res_p = 5.0*micra

###############################################################################
# Cambiar el valor de la resolucion por: res_g o res_m o res_p
###############################################################################

resolution = res_p

###############################################################################
# Brincar hasta la li­nea 121 para editar el eje y las graficas
###############################################################################

c = int(len(matrix[0])/2) # centro de las imagenes

num_submuestras = int(len(matrix[0])/2)
print('Submuestras por slice: ', num_submuestras)
num_slices = len(matrix)
print('\nCantidad de slices: ', num_slices, '\n')

# Funcion que calcula la porosidad a diferentes tamanos
@njit(parallel=True)
def porosity(img, i):
    # scale = int(delta*i/2)
    # definimos el roi (region of interest) que se va a ir incrementando en cada iteraciÃƒÂ³n 
    roi = img[c-i:c+i, c-i:c+i]
    
    # Calculamos la porosidad en cada slice recortado
    # 255 = negro (poro) y 0 = blanco (roca) en caso de imagenes png de avizo
    porosidad= (np.sum(roi==255)/(np.sum(roi==255)+np.sum(roi==0)))*100
    
    return porosidad

# Creamos un arreglo donde guardaremos los valores obtenidos de la porosidad de cada imagen
porosidad_x_slice = []

for x in file_names:
    
    # Leemos cada imagen de la carpeta que contiene todas las imagenes
    image_path = input_images_path + "/" + x
    
    # Convertimos cada imagen a escala de grises
    image = cv2.imread(image_path, 0)
    # image = cv2.imread(image_path)
    
    # Obtenemos el valor de cada umbral con el metodo de Otsu
    # umbral, img_bin = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    
    for i in range(num_submuestras):
        
        # porosidad_x_slice.append(porosity(img_bin, i+1))
        porosidad_x_slice.append(porosity(image, i+1))

# creamos una funcion para dividir una lista en listas del mismo tamano
#@njit(parallel=True)
def div_Lista(lista, size):
    return [lista[n:n+size] for n in range(0,len(lista), size)]

# dividimos la lista "porosidad_x_slice" en listas del mismo tamano
serie_poro = div_Lista(porosidad_x_slice, num_submuestras)

##############################################################
# Registramos el tiempo de finalizacion
end_time = time.time()

# calculamos la diferencia de tiempo
execution_time = end_time - start_time
print(f'Tiempo de ejecucion: {round(execution_time, 2)} segundos')

##############################################################
# Guardamos los datos generados
np.savetxt(ruta_nueva_carpeta + '\porosidades2D.txt', serie_poro)
np.savetxt(ruta_nueva_carpeta + '\porosidades2D.csv', serie_poro)

np.savetxt(ruta_nueva_carpeta + '\dimensiones_experimentos.txt', np.shape(serie_poro))
np.savetxt(ruta_nueva_carpeta + '\dimensiones_tensor.txt', np.shape(matrix))

# np.savetxt(ruta_nueva_carpeta + '\porosidad_total.txt', 'phi_t = ', porosidad_total)
##############################################################

# Para graficar los valores obtenidos, convertimos la lista dividida en 
# listas en un arreglo de numpy y lo trasponemos T
serie_poro_2 = np.asarray(serie_poro).T

porosidad_total = np.mean(serie_poro_2[-1])
print('\nPorosidad total = ', porosidad_total, '%')

# definimos los valores que tendra el eje X
valores_x = []

for m in range(num_submuestras):
    
    ###########################################################################
    # Cambiar el exponente final de area_scaling por 1, 2 o 3
    ###########################################################################
    
    area_scaling = ((m+1)*2*(resolution))**1
    valores_x.append(area_scaling)
    
###########################################################################
# Cambiar las etiquetas y el titulo deacuerdo a la muestra analizada
###########################################################################

# Porosidad experimental
porosidad_exp = 23.0 #45.27163916
vol_core = 20.5695

t = np.arange(0.0, vol_core, 0.001)
# s = t*porosidad_exp/t
t_p = np.arange(0.0, vol_core, 0.001)
# p = t_p*porosidad_total/t_p

x1 = max_pixeles*resolution + 0.01
x2 = vol_core - 0.01
x3 = vol_core + 0.01

fig = plt.figure(figsize=(10, 5))

# Obtener las dimensiones de la matriz
filas, columnas = serie_poro_2.shape

# Graficar los valores de la matriz
for n in range(columnas):
    
    # Definir el color para cada recta utilizando colormap
    cmap = plt.colormaps.get_cmap('jet')
    norm = colors.Normalize(vmin=1, vmax=columnas)
    color = cmap(norm(n))
    
    plt.plot(valores_x, serie_poro_2[:, n], color = color, linewidth = 0.25)

# Agregar un colorbar a la derecha
sm = plt.cm.ScalarMappable(cmap='jet', norm = plt.Normalize(vmin=1, vmax=columnas))
sm.set_array([])
cbaxes = fig.add_axes([1.0, 0.1, 0.03, 0.8])
plt.colorbar(sm, cax = cbaxes, label = 'slice number', location = 'right', pad=0.25)
 