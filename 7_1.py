import numpy as np
from scipy.linalg import solve_triangular
import math


def gmres(A, b, x_0, m, tol):
    n = len(b)
    xs = np.zeros((n, m + 1))  # множество решений x. выше координата - выше точность
    H_m = np.zeros((m + 1, m))  # базис h

    r_0 = b - np.dot(A, x_0)  # начальная невязка
    beta = np.linalg.norm(r_0)  # коэффициент нормы невязки
    xs[:, 0] = r_0 / beta  # v1

    for k in range(m):
        # Arnoldi процесс
        w = np.dot(A, xs[:, k])  # w = A * vj
        for j in range(k + 1):
            H_m[j, k] = np.dot(xs[:, j], w)
            w -= H_m[j, k] * xs[:, j]

        H_m[k + 1, k] = np.linalg.norm(w)

        # брейкер из методички
        #if H_m[k + 1, k] == 0:


        xs[:, k + 1] = w / H_m[k + 1, k]

        # Решение подзадачи минимизации
        y, _, _, _ = np.linalg.lstsq(H_m[:k + 1, :k], beta * np.eye(k + 1), rcond=None)
        #y = solve_triangular(H_m[:k + 1, :k], beta * np.eye(k + 1), lower=True)

        # Обновление приближенного решения
        x_k = x_0 + np.dot(xs[:, :k + 1], y)
        r_k = b - np.dot(A, x_k)

        # Проверка критерия сходимости
        if np.linalg.norm(r_k) < tol:
            return x_k

    # Возврат наилучшего приближенного решения
    return x_k


def gmresMetod(A, b, x_0):
    n = len(b)
    xs = np.zeros((n, n * 10))  # множество решений x. выше координата - выше точность
    H_m = np.zeros((n, n))  # базис h

    r_0 = b - np.dot(A, x_0)  # начальная невязка
    beta = np.linalg.norm(r_0)  # коэффициент нормы невязки
    xs[:, 0] = r_0 / beta  # v1

    for k in range(m):
        # Arnoldi процесс
        w = np.dot(A, xs[:, k])  # w = A * vj
        for j in range(k + 1):
            H_m[j, k] = np.dot(xs[:, j], w)
            w -= H_m[j, k] * xs[:, j]

        H_m[k + 1, k] = np.linalg.norm(w)

        # брейкер из методички
        #if H_m[k + 1, k] == 0:


        xs[:, k + 1] = w / H_m[k + 1, k]


def f(x):
    return math.sin(x)

def realize():
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
        #ans = gmres(A, b, )
        break


def test():
    A = np.array([[1, 1], [2, 6]])
    b = np.array([3, -4])

    # Начальное приближение
    x_0 = np.zeros_like(b)

    # Число итераций
    m = 100

    # Точность
    tol = 1e-6

    # Вызов GMRES
    result = gmres(A, b, x_0, m, tol)

    print("Приближенное решение:", result)

    
test()