# comment 02.12 (0:38)
# test сейчас работает только в частном случае, если немного изменить входные условия,
# например взять первое уравнение 1, 7 - система умрёт, будет давать слишком
# маленькие значения. есть подозрение, что проблема в методе, а именно в ебучем
# плоском вращении. нужно до конца определиться с индексами, к которым мы применяем
# это самое вращение, потому что есть подозрение, что перенос из методички индексов,
# где каждый должен быть на 1 меньше, не увенчался успехом

import numpy as np
from scipy.linalg import solve_triangular
import math


def givenRotation(a, b):
    """
    Вычисляет коэффициенты c и s для плоского вращения Гивенса.

    Параметры:
    a, b: Элементы, которые нужно обнулить (второй элемент b).

    Возвращает:
    c, s: Коэффициенты плоского вращения.
    """
    r = np.hypot(a, b)
    c = a / r
    s = b / r
    #print(c, s)
    return c, s

def gmresMetod(A, b, x0, eps):
    over = False
    p = 0
    n = len(b)
    h = np.zeros((n + 1, n))
    e1 = np.zeros(n + 1)
    e1[0] = 1
    r0 = b - A @ x0
    beta = np.linalg.norm(r0)
    g = beta * e1
    v = np.zeros((n + 1, n))
    v[0] = r0 / beta
    for j in range(n):
        w = A @ v[j]
        for i in range(j + 1):
            h[i, j] = w @ v[i]
            w = w - h[i, j] * v[i]
        h[j + 1, j] = np.linalg.norm(w)
        if h[j + 1, j] == 0:
            p = j
            break
        v[j + 1] = w / h[j + 1, j]
        for i in range(j):
            ci, si = givenRotation(h[i, j], h[i + 1, j])
            temp1 = ci * h[i, j] + si * h[i + 1, j]
            temp2 = -si * h[i, j] + ci * h[i + 1, j]
            h[i, j] = temp1
            h[i + 1, j] = temp2

        cj, sj = givenRotation(h[j, j], h[j + 1, j])
        temp1 = cj * h[j, j] + sj * h[j + 1, j]
        temp2 = -sj * h[j, j] + cj * h[j + 1, j]
        h[j, j] = temp1
        h[j + 1, j] = temp2

        temp1 = cj * g[j] + sj * g[j + 1]
        temp2 = -sj * g[j] + cj * g[j + 1]
        g[j] = temp1
        g[j + 1] = temp2

        if abs(g[j + 1]) < eps:
            over = True
            p = j
            break

    y = np.linalg.solve(h[:p + 1, :p + 1], g[:p + 1])
    dd = np.dot(v[:p + 1].T, y)

    x = x0 + dd
    return x, over


def f(x):
    return math.sin(x)

def realize():
    eps = 10**(-4)
    for i in range(6, 10):
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
        print(A)
        print(b)
        x_exact = np.linalg.solve(A, b)
        print("Точное решение:", x_exact)
        ans, metodOver = gmresMetod(A, b, b, eps)
        max_iter = 50000
        iter = 0
        while not metodOver and iter < max_iter:
            ans, metodOver = gmresMetod(A, b, b, eps)
            iter += 1
        print(n, iter, ans)
        print("diffs:")
        for i in range(len(A)):
            print(abs(ans[i] - x_exact[i]))
        break


def test():
    A = np.array([[1, 1], [2, 6]])
    b = np.array([3, -4])

    # Точное решение
    x_exact = np.linalg.solve(A, b)
    print("Точное решение:", x_exact)

    # Начальное приближение
    x_0 = np.zeros_like(b)
    ans, metodOver = gmresMetod(A, b, x_0, 10**(-4))
    max_iter = 50000
    iter = 0
    while not metodOver and iter < max_iter:
        ans, metodOver = gmresMetod(A, b, ans, 10**(-4))
        iter += 1
    print(2, iter, ans)
    print("diffs:")
    for i in range(len(A)):
        print(abs(ans[i] - x_exact[i]))



realize()

# main 10
# побочные 1
# x 1 2 3 4 5...
# b = Ax
