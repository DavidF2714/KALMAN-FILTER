import numpy as np

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

F_t = np.array([[1, 1], [0, 1]])
Q_t = np.array([[0, 0], [0, 2]])
S_t = np.array([[1, 0], [0, 1]])


S_t = givens_rotation(F_t, Q_t, S_t)
print(S_t)