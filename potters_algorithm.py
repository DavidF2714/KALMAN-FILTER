"""
H es matriz de transición
yi es matriz de covarianza
Ri varianza de medida (suposición es diagonal)


1.- Xpredicción t Spredicción t
2.- ciclo 1 a tamaño de y donde agarramos row de matriz de transición elemento de y(matriz de covarianza), Ri varianza de medida i
2.1.- Encontrar S después de que la medida i ha sido procesada. Oi = Sti-1,t Hti  ai= 1/0ti 0i + Ri  alphai = ai/1+sqrt(aiRi) Si,t=Si-1(I-aalphai= Oi 
2.2.- Ki,t = Si,t Oi kamlan gain
2.3.- Xi,t = xi-1+Ki,t(yi,-hixi-1,t)
3 St = St,Sn,t	Xt = Xn,t
XHR son matrices

"""
#Kalman filter
import numpy as np

def Potter(X,S,y,H,R):
    #inicializar X0,t Xpt y So,t Spt
    n = len(y)
    x_t = X
    Si = S

    for i in range(n):
        Hi = H[i,:]
        yi = y[i]
        Ri = R[i,i] #varianza

        phi = np.dot(Si,Hi.T)
        ai = 1/(np.dot(phi.T,phi) + Ri)
        gammai = ai / (1 + np.sqrt(ai * Ri))

        Si = Si *(np.eye(Si.shape[0]) - ai * gammai * np.dot(phi,phi.T))

        Ki = np.dot(Si,phi)

        x_t = x_t + Ki * (yi- np.dot(Hi,x_t))

    return x_t, Si


# Initial state estimate (e.g., a 2-dimensional vector)
x_p_t = np.array([0, 0])

# Initial covariance estimate (e.g., a 2x2 identity matrix)
S_p_t = np.array([[1, 0], 
                  [0, 1]])

# Measurements (e.g., 2 measurements)
y_t = np.array([1, 2])

# Measurement matrix (2 measurements, 2 state variables)
H = np.array([[1, 0], 
              [0, 1]])

# Measurement noise covariance matrix (2x2 diagonal matrix)
R = np.array([[0.1, 0], 
              [0, 0.1]])

# Run the Potter function
x_t, S_t = Potter(x_p_t, S_p_t, y_t, H, R)

print("Updated state estimate (x_t):")
print(x_t)
print("\nUpdated covariance estimate (S_t):")
print(S_t)