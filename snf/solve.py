import numpy as np

def lstsq(Psi, w): # Least squares (min-norm if underdetermined)
    return np.linalg.lstsq(Psi, w, rcond=None)[0]

def omp(Psi, w, tol=1e-3, max_k=None):
    '''
    Orthogonal matching pursuit. Keeps an orthonormal basis Q of the chosen columns (Psi_S = Q Rm) so each step is cheap.

    Psi (np.array): Dimensions (M, K)
    w (np.array): Dimensions (M)
    tol (float): stop when ||residual|| / ||w|| <= tol
    max_k (int): maximum support size

    Returns x; dimensions (K)
    '''
    M, K = Psi.shape
    max_k = min(max_k or M // 2, M)
    norms = np.linalg.norm(Psi, axis=0)
    Q, Rm = np.zeros((M, max_k), complex), np.zeros((max_k, max_k), complex)
    support, r = [], w.copy()
    while np.linalg.norm(r) > tol * np.linalg.norm(w) and len(support) < max_k:
        corr = np.abs(Psi.conj().T @ r) / norms # pick the column most correlated with the residual
        corr[support] = 0
        j, k = int(np.argmax(corr)), len(support)
        q = Psi[:, j].copy()
        for _ in range(2): # Gram-Schmidt, twice for stability
            c = Q[:, :k].conj().T @ q
            q -= Q[:, :k] @ c
            Rm[:k, k] += c
        Rm[k, k] = np.linalg.norm(q)
        Q[:, k] = q / Rm[k, k]
        support.append(j)
        r -= Q[:, k] * (Q[:, k].conj() @ r)
    k = len(support)
    x = np.zeros(K, complex)
    x[support] = np.linalg.solve(Rm[:k, :k], Q[:, :k].conj().T @ w) if k else []
    return x
