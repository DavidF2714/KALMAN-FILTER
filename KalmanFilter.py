import numpy as np
import pandas as pd
import math
from scipy.linalg import ldl

# 1. Load the CSV file
df = pd.read_csv('User1_Pre2.csv', delimiter=',')

# 2. Select the significant sensors
sensores_significativos = ['F4', 'F8', 'AF4']
datos = df[sensores_significativos].values

# Parameters
n_ensemble = 50  # Size of the ensemble
n_sensors = len(sensores_significativos)
frecuencia_muestreo = 128
dt = 1 / frecuencia_muestreo  # Time step
n_steps = len(datos)  # Number of time steps
y = datos  # Observations
R = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])  # Observation noise covariance

# Initialize ensemble members (random perturbations around initial state x_0)
x_0 = datos[0]
ensemble = np.array([x_0 + np.random.normal(0, 0.1, n_sensors) for _ in range(n_ensemble)])

def series_taylor(dt, n=2):
    """Calculate the transition matrix F using Taylor series expansion."""
    F = np.eye(n_sensors)  # Identity matrix
    # Adjust the matrix exponentiation for your specific system dynamics
    for i in range(1, n + 1):
        F += (dt ** i) / math.factorial(i) * np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) ** i
    return F

def givens_rotation(F, Q, S):
    """Apply Givens rotation to the covariance matrix S."""
    m = S.shape[1]
    U = np.block([[F.T @ S.T], [np.sqrt(Q).T]])
    
    for j in range(1, m + 1):
        for i in range(2 * m - 1, j - 1, -1):
            B = np.eye(2 * m)
            a = U[i - 1, j - 1]
            b = U[i, j - 1]
            if b == 0:
                c = 1
                s = 0
            else:
                if abs(b) > abs(a):
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

def potterAlg(S_p, H, R, ensemble, y_t):
    """Potter algorithm to update state and covariance with Kalman gain."""
    n_ensemble = ensemble.shape[0]
    H_i = H
    y_i = y_t

    # Compute ensemble mean
    x_mean = np.mean(ensemble, axis=0)

    # Compute Kalman gain
    S_pH_T = S_p @ H_i.T
    PH_T = np.cov(ensemble.T) @ H_i.T
    S_pH_T_inv = np.linalg.inv(S_pH_T @ H_i + R)
    K_t = PH_T @ S_pH_T_inv

    # Update each ensemble member
    for i in range(n_ensemble):
        ensemble[i] = ensemble[i] + K_t @ (y_i - H_i @ ensemble[i])

    # Update covariance
    S_t = S_p - K_t @ H_i @ S_p

    return ensemble, S_t

def EnKF(dt, ensemble, P_0, H, R, y, n_steps):
    """Ensemble Kalman Filter implementation."""
    L, D, perm = ldl(P_0)
    S_t = L @ np.diag(np.sqrt(np.diag(D)))
    
    for step in range(n_steps):
        # 1. Calculate the transition matrix F
        F = series_taylor(dt)

        # 2. Apply Givens rotation
        S_t = givens_rotation(F, np.eye(S_t.shape[0]), S_t)

        # 3. Use Potter algorithm to update state and covariance
        ensemble, S_t = potterAlg(S_t, H, R, ensemble, y[step])

    # Compute final ensemble mean and covariance
    x_final = np.mean(ensemble, axis=0)
    P_final = np.cov(ensemble.T)

    return x_final, P_final

# Identity matrix for observations
H = np.eye(n_sensors)

# Initial covariance matrix (identity)
P_0 = np.eye(n_sensors)

# Run the Ensemble Kalman Filter
x_final, P_final = EnKF(dt, ensemble, P_0, H, R, y, n_steps)

print("Final state estimate (ensemble mean):", x_final)
print("Final covariance estimate:", P_final)