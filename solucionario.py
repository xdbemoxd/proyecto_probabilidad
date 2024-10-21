import numpy as np
from scipy.stats import binom
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

def cantidad_elemtos(arr):

    aux = 0

    for i in arr:
        if i == 6:
            aux=+1
    
    return aux

def numeros_pares(arr):
    
    aux = []

    for i in arr:
        if i%2==0:
            aux.append(i)

    return aux

def numeros_impares(arr):
    aux = []

    for i in arr:
        if i%2==1:
            aux.append(i)

    return aux

def lectura_1():
    print('Considerando un lanzamiento de un dado y considerando los siguientes eventos aleatorios:')

    print('A = {el resultado del lanzamiento de un dado es 6}')

    print('B = {el resultado del lanzamiento de un dado es par}')

    print('C = {el resultado del lanzamiento de un dado es impar}')

    print("""calcula las siguientes probabilidades:
          P(A|B)=?
          P(A|C)=?
          P(B|C)=?""")
    
    elementos = np.arange(1,7)

    p_a_b = cantidad_elemtos( numeros_pares( elementos ) ) / 6

    p_a_c = cantidad_elemtos( numeros_impares( elementos ) ) / 6

    p_b_c = len( numeros_pares( numeros_impares( elementos ) ) ) / 6

    print("""los resultados son los siguientes:
          P(A|B)={}
          P(A|C)={}
          P(B|C)={}""".format( p_a_b, p_a_c, p_b_c ) )

def elementos_comunes(a,b):

    arr = []

    for i in a:
        if i in b:
            arr.append(i)

    return arr

def calculo(p_a,p_b,elementos):
    cuenta = ( len( elementos_comunes( p_a, p_b ) ) / len( elementos ) ) / ( len( p_b ) / len( elementos ) )
    return cuenta

def lectura_2():

    print("""Considerando una ruleta de doce números {1,2,3,4,5,6,7,8,9,10,11,12}
          dos jugadores eligen 6 números, cada uno de ellos. Supón que el jugador 1 elige A = {1,2,3,4,5,6} y calcula las siguientes probabilidades:
          P(A|B) sabiendo que el jugador 2 elige B = {2,4,6,8,10,12}
          P(A|B) sabiendo que el jugador 2 elige B = {1,3,5,7,9,11}
          P(A|B) sabiendo que el jugador 2 elige B = {5,6,7,8,9,10}""")
    elementos = np.arange(1,13)

    p_a = np.array([1,2,3,4,5,6])

    p_b = np.array([2,4,6,8,10,12])

    arreglo = []

    arreglo.append( calculo( p_a, p_b,elementos ) )

    p_b = np.array([1,3,5,7,9,11])

    arreglo.append( calculo( p_a, p_b,elementos ) )

    p_b = np.array([5,6,7,8,9,10])

    arreglo.append( calculo( p_a, p_b,elementos ) )

    print("""los resultados son las siguientes probabilidades:
          P(A|B)={} B = 2,4,6,8,10,12
          P(A|B)={} B = 1,3,5,7,9,11
          P(A|B)={} B = 5,6,7,8,9,10""".format( arreglo[0], arreglo[1], arreglo[2] ) )

def lectura_3():
    print("""Considera un problema donde se lanzan dos monedas, sean m1 y m2. Verifica la regla del producto para las siguientes probabilidades 
          (dibuja el espacio muestral y calcula cada probabilidad por separado):
          P(m1=cara,m2=sello)
          P(m1=cara|m2=sello)
          P(m2=sello)""")
    
    print(""" 
      [m1.m2]
      [cara,cara]
      [cara,sello]
      [sello,cara]
      [sello,sello] """)
    
    p_m1_m2 = 1/4
    p_b = 1/2

    p_m1_m2_2 = p_m1_m2 * p_b

    print("""La solucion es la siguiente
          P(m1=cara,m2=sello)={}
          P(m1=cara|m2=sello)={}
          P(m2=sello)={}
      """.format( p_m1_m2_2, p_m1_m2, p_b ) )

def primer_bloque():

    listo = False

    while( not listo ):

        print("""Ingrese la lectura que desea ver resuelta, junto a su enunciado
          1)Primera lectura
          2)Segunda lectura
          3)Tercera lectura
          4)salir""")
        
        opc = int( input( 'ingrese el numero: ' ) )

        if opc == 1:
            lectura_1()
        elif opc == 2:
            lectura_2()
        elif opc == 3:
            lectura_3()
        elif opc == 4:
            listo = True
        
        print('\n')


def segundo_bloque():
    
    print("""Calcula a mano las siguientes probabilidades (tomando p=0.5, por lo tanto 1−p=0.5):
          Probabilidad de obtener 3 caras a partir de 12 lanzamientos de moneda.
          Probabilidad de obtener 5 o menos caras a partir de 10 lanzamientos de moneda.
          Probabilidad de obtener menos de 6 caras a partir de 10 lanzamientos de moneda.
          Calcula las mismas probabilidades anteriores pero considerando ahora p=0.3.""")
    
    p = 0.5

    n = 12

    k = 3

    reuslt_teorico_1 = binom(n,p).pmf(k)

    p = 0.5

    n = 10

    k = 5

    reuslt_teorico_2 = binom(n,p).cdf(k)

    p = 0.5

    n = 10

    k = 6 - 1

    reuslt_teorico_3 = binom(n,p).cdf(k)
    
    print("\n")
    print("""teniendo en cuanta que p=0.5, los resultados seran los siguiente: 
          Probabilidad de obtener 3 caras a partir de 12 lanzamientos de moneda={}
          Probabilidad de obtener 5 o menos caras a partir de 10 lanzamientos de moneda={}
          Probabilidad de obtener menos de 6 caras a partir de 10 lanzamientos de moneda={}
          """.format(reuslt_teorico_1,reuslt_teorico_2,reuslt_teorico_3) )
    
    p = 0.3

    n = 12

    k = 3

    reuslt_teorico_1 = binom(n,p).pmf(k)

    p = 0.3

    n = 10

    k = 5

    reuslt_teorico_2 = binom(n,p).cdf(k)

    p = 0.3

    n = 10

    k = 6 - 1

    reuslt_teorico_3 = binom(n,p).cdf(k)

    print("""teniendo en cuanta que p=0.3, los resultados seran los siguiente: 
          Probabilidad de obtener 3 caras a partir de 12 lanzamientos de moneda={}
          Probabilidad de obtener 5 o menos caras a partir de 10 lanzamientos de moneda={}
          Probabilidad de obtener menos de 6 caras a partir de 10 lanzamientos de moneda={}
          """.format(reuslt_teorico_1,reuslt_teorico_2,reuslt_teorico_3) )
    
    print('\n')


def optimal_mu(arr):

    sum=0

    for i in arr:
        sum+=i
    
    return sum/len(arr)

def optimal_sigma(arr):

    mu = arr.mean()
    sum = 0
    y = mu**2
    for i in arr:
        x = i**2
        xy = -(2*i*mu)
        z = x+xy+y
        sum+=z
    
    sum = sum/len(arr)

    return sum

def tercer_bloque():
    print("""Comprobación numérica Vamos ahora a hacer una comprobación numérica de que esos parámetros efectivamente ajustan de manera óptima los datos. Construye funciones en Python que te permitan calcular directamente los parámetros óptimos según las ecuaciones encontradas:
          μ=1n∑inxi
          σ2=1n∑in(xi−μ)2 """)
    
    print("Usare el data frame de cars.csv, usando la columna precio")

    print("usare la funcion de la libreria panadas para ambos casos y las comprobare con una funcion propia")

    df = pd.read_csv("cars.csv")

    arr = df['price_usd']

    print('Mu usando una funcion propia:{}, usando la funcion de pandas:{}'.format( optimal_mu( arr ), arr.mean() ) )

    print('Sigma usando funcion propia:{}, usando la libreria de pandas:{}'.format( optimal_sigma( arr ), arr.std() ) )

    print('revisa entre los valores 0 y 8000, se hace una curva normal')

    values, dist = np.unique(arr, return_counts=True)
    plt.bar(values, dist/len(arr)) 

    dist = norm( optimal_mu( arr ), arr.std() )
    x = np.arange( 0, 50000, 1000 )
    y = [ dist.pdf( value ) for value in x ]
    plt.plot( x, y )

    plt.show()

def cuarto_bloque():
    pass

def menu():
    listo = False
    while( not listo ):
        print("""Ingrese la opción de preferencia:
          1)lecturas 1-3
          2)lecturas 6-8
          3)lecturas 11-12
          4)salir""")
    
        opc = int( input( 'ingrese el numero: ' ) )

        if opc == 1:
            primer_bloque()
        elif opc == 2:
            segundo_bloque()
        elif opc == 3:
            tercer_bloque()     
        elif opc == 4:
            listo = True
            
menu()