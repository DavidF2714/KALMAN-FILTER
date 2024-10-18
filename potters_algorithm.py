import numpy as np

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


# Prueba con valores de ejemplo
S = np.array([[1, 0], [0, 1]])  # Matriz de covarianza inicial
H = np.array([[1, 0], [0, 1]])  # Matriz de observaciones
R = np.array([[0.1, 0], [0, 0.1]])  # Matriz de ruido
x = np.array([0, 0])  # Estado inicial
y = np.array([1, 2])  # Observaciones

# Ejecutamos el algoritmo de Potter
x_i, s_i = potterAlg(S, H, R, x, y)
print(x_i)
print(s_i)
