import numpy as np
import pandas as pd
import math

# 1. Cargar el archivo CSV
df = pd.read_csv('User1_Pre2.csv', delimiter=',')  # Asegúrate de usar el delimitador correcto

# 2. Seleccionar los sensores significativos
sensores_significativos = ['F4', 'F8', 'AF4']
datos = df[sensores_significativos].values

# 3. Preprocesar los datos
# Puedes normalizar o estandarizar los datos si es necesario
# En este ejemplo, simplemente tomaremos el primer registro como estado inicial
x_0 = datos[0]  # Estado inicial
P_0 = np.eye(len(sensores_significativos))  # Matriz de covarianza inicial (identidad)
H = np.eye(len(sensores_significativos))  # Matriz de observaciones (identidad)
R = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])  # Matriz de ruido (ajusta según tu caso)

# Frecuencia de muestreo
frecuencia_muestreo = 128
dt = 1 / frecuencia_muestreo  # Paso de tiempo

# 4. Ejecutar el Filtro de Kalman Extendido
n_steps = len(datos)  # Número de pasos de tiempo (3073)
y = datos  # Observaciones

def potterAlg(S_p, H, R, x_p, y_t):
    
    x_t = x_p
    S_t = S_p
    n = len(y_t)
    
    for i in range(n):
        H_i=H[i,:]
        y_i = y_t[i]
        R_i = R[i,i]
        
        phi = np.dot(S_t, H_i.T)
        a_i = 1/(np.dot(phi.T, phi) + R_i)
        
        gamma = a_i/(1+np.sqrt(a_i*R_i))

        S_t = S_t * (np.eye(S_t.shape[0]) - a_i * gamma * np.dot(phi, phi.T))
        
        K_t = np.dot(S_t, phi)
        
        x_t = x_t + K_t * (y_i - np.dot(H_i, x_t))      
        
    return x_t, S_t

def givens_rotation(F, Q, S):
    m = S.shape[1]

    U = np.block([
        [F.T @ S.T],
        [np.sqrt(Q).T] 
    ])
    
    for j in range(1, m + 1):
        for i in range(2 * m - 1, j - 1, -1): 
            B = np.eye(2 * m)
            a = U[i - 1, j - 1]
            b = U[i, j - 1]
            
            if b == 0:
                c = 1
                s = 0
            else:
                if np.abs(b) > np.abs(a):
                    r = a / b
                    s = 1 / np.sqrt(1 + r ** 2)
                    c = s * r
                else:
                    r = b / a
                    c = 1 / np.sqrt(1 + r ** 2)
                    s = c * r
                    

            B[i - 1:i + 1, i - 1:i + 1] = np.array([[c, -s], [s, c]])
            U = B.T @ U


    S_t = U[:m, :]
    return S_t


# Implementar el EnKF
def series_taylor(dt, n=2):
    """Calcula la matriz de transición F usando Series de Taylor."""
    F = np.eye(len(sensores_significativos))  # Matriz identidad
    for i in range(1, n + 1):
        F += (dt ** i) / math.factorial(i) * np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) ** i  # Ajusta para tu sistema
    return F

def LDLT_decomposition(P):
    """Convierte la matriz de covarianza P en matriz de raíz cuadrada S usando descomposición LDLT."""
    L = np.linalg.cholesky(P)
    return L

def EnKF(dt, x_0, P_0, H, R, y, n_steps):
    """Implementación del Filtro de Kalman Extendido."""
    x_t = x_0
    S_t = LDLT_decomposition(P_0)
    
    for step in range(n_steps):
        # 1. Calcular la matriz de transición F
        F = series_taylor(dt)

        # 2. Aplicar la Rotación de Givens
        S_t = givens_rotation(F, np.eye(S_t.shape[0]), S_t)

        # 3. Algoritmo de Potter para actualizar el estado y la covarianza
        x_t, S_t = potterAlg(S_t, H, R, x_t, y[step])

    return x_t, S_t

# Ejecutar el Filtro de Kalman Extendido
x_final, S_final = EnKF(dt, x_0, P_0, H, R, y, n_steps)

print("Estado final:", x_final)
print("Covarianza final:", S_final)
