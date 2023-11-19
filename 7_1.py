import numpy as np
from scipy.linalg import solve_triangular
import math


def gmres(A, b, x_0, m, tol):
    n = len(b)
    Q_m = np.zeros((n, m + 1))
    H_m = np.zeros((m + 1, m))

    r_0 = b - np.dot(A, x_0)
    beta = np.linalg.norm(r_0)
    Q_m[:, 0] = r_0 / beta

    for k in range(m):
        # Arnoldi процесс
        v = np.dot(A, Q_m[:, k])  # w = A * vj
        for j in range(k + 1):
            H_m[j, k] = np.dot(Q_m[:, j], v)
            v -= H_m[j, k] * Q_m[:, j]

        H_m[k + 1, k] = np.linalg.norm(v)
        Q_m[:, k + 1] = v / H_m[k + 1, k]

        # Решение подзадачи минимизации
        y = solve_triangular(H_m[:k + 1, :k], beta * np.eye(k + 1), lower=True)

        # Обновление приближенного решения
        x_k = x_0 + np.dot(Q_m[:, :k + 1], y)
        r_k = b - np.dot(A, x_k)

        # Проверка критерия сходимости
        if np.linalg.norm(r_k) < tol:
            return x_k

    # Возврат наилучшего приближенного решения
    return x_k


def f(x):
    return math.sin(x)


eps = 10**(-4)
print(math.pi / 4)
for i in range(6, 20):
    h = math.pi / (i * 4)
    n = i
    xs = np.linspace(0, math.pi / 4, n + 1)
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    for j in range(1, n):
        A[j - 1, j - 1] = -1 / h ** 2
        A[j - 1, j + 1] = -1 / h ** 2
        A[j - 1, j] = 2 / h ** 2
    A[n - 1, 0] = 1
    A[n, n] = 1
    for j in range(n+1):
        b[j] = math.sin(xs[j])
    #print(A)
    #print(b)
    ans = gmres(A, b, )
    break
