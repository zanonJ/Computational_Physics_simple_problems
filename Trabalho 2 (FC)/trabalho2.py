# -*- coding: utf-8 -*-
"""
Trabalho 2 (Física Computacional) — Circuitos eléctricos e moléculas
=====================================================================

This file joins the old scripts ``Exercício 2 .py``, ``exercicio 4.py``,
``exercicio 5.py``, ``exercício 7.py``, ``exercício 9.py`` and
``exercício 10.py`` into one module. Each exercise is a function, and they can
be run separately.

Part I.1 — Resistor network (exercise 2)
----------------------------------------
Kirchhoff's current law at each node of the circuit (with V = 10 V at node a
and node i grounded) gives a 7×7 linear system A·V = b for the potentials
Vb ... Vh. It is solved by Gaussian elimination without pivoting followed by
back substitution, and the solution is checked by computing the residual A·V - b.

Part I.2 — RLC circuit (exercises 4 and 5)
------------------------------------------
Kirchhoff's voltage law for a series RLC circuit:

    L dI/dt + R I + Q/C = V(t),     I = dQ/dt

This is the equation of a damped, driven harmonic oscillator:
L ↔ mass (inertia), R ↔ friction, 1/C ↔ spring constant, V(t) ↔ external
force, Q ↔ position, I ↔ velocity. The natural frequency is
ω0 = 1/sqrt(LC).

* Exercise 4: free LC circuit (R = 0, V = 0, L = 1 H, C = 0.05 F, Q(0) = 0,
  I(0) = 1 A). Analytic solution: Q(t) = I0/ω0 · sin(ω0 t),
  I(t) = I0 cos(ω0 t) with ω0 = sqrt(20) ≈ 4.47 rad/s. The energy
  Q²/2C + L I²/2 is conserved, so the amplitude must stay constant. This is
  the check that the step h is stable.
* Exercise 5: R = 0.1 Ω and V(t) = 0.1 cos(5t). The free oscillation decays
  with time constant 2L/R = 20 s, and the circuit then follows the driving
  frequency ω = 5 rad/s (steady state). In phase space (Q, I) the orbit
  goes from the initial ellipse to the smaller limit-cycle ellipse.

Both are integrated with a classical 4th-order Runge-Kutta (RK4) for the
coupled first-order system y = (Q, I).

Part II — Vibrations of a linear "molecule" (exercises 7, 9, 10)
----------------------------------------------------------------
Masses on a line joined by springs:

    m1 x1'' = -k1 (x1 - x2)
    m2 x2'' = -k1 (x2 - x1) - k2 (x2 - x3)
    m3 x3'' = -k2 (x3 - x2)

* Exercise 7: RK4 integration for m1 = m3 = 1, m2 = 16, k1 = k2 = 1 and
  four initial conditions. Depending on the initial condition you see pure
  translation of the centre of mass, the symmetric stretch (outer atoms move
  in opposition, the centre stays still), or a mixture of modes.
* Exercise 9: normal modes. Trying x_i = X_i e^{iωt} with equal masses and
  springs gives the eigenvalue problem K·X = (ω/ω0)² X with
  K = [[1,-1,0],[-1,2,-1],[0,-1,1]]. The eigenvalues are 0 (translation),
  1 (symmetric stretch, centre fixed) and 3 (antisymmetric stretch). They
  are found with the Jacobi rotation method.
* Exercise 10: the same for a chain of 9 masses and 8 springs (tridiagonal
  matrix). The eigenvalues are ω²/ω0² = 2 - 2cos(nπ/9) = 4 sin²(nπ/18),
  n = 0..8. The 4 lowest modes look like standing waves with 0, 1, 2 and 3
  nodes.

Usage
-----
    python trabalho2.py            # run every exercise
    python trabalho2.py 4 5        # run only exercises 4 and 5
    python trabalho2.py 7 --ci 2   # exercise 7 only with initial condition 2
"""
import sys

import numpy as np
import pandas
import matplotlib.pyplot as plt


# =============================================================================
# Helper functions
# =============================================================================

def matrix_append(n):
    """Creates an n×n list of lists (filled with empty lists)."""
    M = []
    for i in range(0, n):
        M.append([])
        for j in range(0, n):
            M[i].append([])
    return M


def vector_append(n):
    M = []
    for i in range(0, n):
        M.append([])
    return M


def matrix_identity(n):
    M = matrix_append(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 1.0
            else:
                M[i][j] = 0.0
    return M


def rk4(derivadas, y0, t0, tf, N):
    """Classical Runge-Kutta 4 for a system dy/dt = derivadas(t, y).

    All components of y advance together at each stage (k1..k4), which is
    needed for coupled equations like Q' = I, I' = -Q/LC.
    Returns the arrays t (N+1) and y (N+1, len(y0)).
    """
    h = (tf - t0)/N
    t = t0 + h*np.arange(N + 1)
    y = np.zeros((N + 1, len(y0)))
    y[0] = y0
    for n in range(N):
        k1 = derivadas(t[n], y[n])
        k2 = derivadas(t[n] + h/2.0, y[n] + h*k1/2.0)
        k3 = derivadas(t[n] + h/2.0, y[n] + h*k2/2.0)
        k4 = derivadas(t[n] + h, y[n] + h*k3)
        y[n + 1] = y[n] + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return t, y


# =============================================================================
# Part I.1 — Resistor network
# =============================================================================

def E_G_sem_pivoting(A, b, n):
    """Gaussian elimination without pivoting: turns A into an upper triangular
    U (and b into c), then solves U x = c by back substitution.
    Note: A and b are modified in place."""
    m = matrix_append(n)      # multipliers
    for k in range(0, n):
        for i in range(k+1, n):
            m[i][k] = -(A[i][k]/A[k][k])
            A[i][k] = 0
            b[i][0] = b[i][0] + m[i][k]*b[k][0]

            for j in range(k+1, n):
                A[i][j] = A[i][j] + m[i][k]*A[k][j]

    x1 = vector_append(n)           # back substitution for U x = c
    n1 = n-1
    x1[n1] = b[n1][0]/A[n1][n1]
    for k in range(n1, -1, -1):
        soma = 0.0
        for j in range(k+1, n):
            soma += A[k][j]*x1[j]
        x1[k] = (1.0/A[k][k])*(b[k][0] - soma)
    return x1


def exercicio2():
    """Potentials Vb..Vh of the circuit (Kirchhoff nodes, V = 10 V)."""
    # Matrizes obtidas do circuito
    A = [[3.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
         [0.0, 3.0, 0.0, -1.0, -1.0, 0.0, 0.0],
         [2.0, 0.0, -3.0, 0.0, 0.0, 1.0, 0.0],
         [2.0, 1.0, 0.0, -6.0, 0.0, 1.0, 2.0],
         [0.0, 1.0, 0.0, 0.0, -3.0, 0.0, 2.0],
         [0.0, 0.0, 1.0, 1.0, 0.0, -3.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, -3.0]]
    b = [[10.0], [10.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
    A0, b0 = np.array(A), np.array(b).ravel()   # copies for the check
    n = len(A)

    x = E_G_sem_pivoting(A, b, n)
    V = ["Vb =", "Vc =", "Vd =", "Ve =", "Vf =", "Vg =", "Vh ="]
    print(pandas.DataFrame(x, V, ["volts(V)"]))
    print("Verificação: max|A·V - b| =", np.max(np.abs(A0 @ np.array(x) - b0)))


# =============================================================================
# Part I.2 — RLC circuit
# =============================================================================

def exercicio4(N=12500):
    """Free LC circuit: R = 0, V = 0, L = 1 H, C = 0.05 F, Q(0) = 0, I(0) = 1 A.
    The numerical result is compared with the analytic solution."""
    L = 1.0
    C = 0.05

    def derivadas(t, y):
        Q, I = y
        return np.array([I, -Q/(L*C)])      # dQ/dt = I ; dI/dt = -Q/(LC)

    t0, tf = 0.0, 125.0       # Tempo inicial e final em segundos
    t, y = rk4(derivadas, [0.0, 1.0], t0, tf, N)
    Q, I = y[:, 0], y[:, 1]

    w0 = 1/np.sqrt(L*C)
    Q_exato = (1.0/w0)*np.sin(w0*t)
    print("Passo:", (tf-t0)/N)
    print("Frequência natural w0 = 1/sqrt(LC) =", w0, "rad/s")
    print("Erro máximo em Q(t) face à solução analítica:", np.max(np.abs(Q - Q_exato)))
    energia = Q**2/(2*C) + L*I**2/2
    print("Variação relativa da energia Q²/2C + LI²/2:", (energia.max()-energia.min())/energia[0])

    plt.figure()
    plt.plot(t, Q, label='RK4')
    plt.plot(t, Q_exato, '--', label='analítica')
    plt.ylabel('Q(t)')
    plt.xlabel('tempo (t)')
    plt.xlim(0.0, 125.0)
    plt.title('Carga: Q(t) ')
    plt.legend()

    plt.figure()
    plt.plot(t, I)
    plt.ylabel('I(t)')
    plt.xlabel('tempo (t)')
    plt.xlim(0.0, 125.0)
    plt.title('Corrente: I(t)')
    plt.show()


def exercicio5(N=25000):
    """Damped and driven RLC: R = 0.1 Ω, V(t) = 0.1 cos(5t)."""
    L = 1.0
    C = 0.05
    R = 0.1

    def derivadas(t, y):
        Q, I = y
        dI = (-Q/(L*C)) - (I*R)/L + ((0.1)*np.cos(5*t))/L
        return np.array([I, dI])

    t0, tf = 0.0, 125.0
    t, y = rk4(derivadas, [0.0, 1.0], t0, tf, N)
    Q, I = y[:, 0], y[:, 1]
    print("Passo:", (tf-t0)/N)
    print("Tempo de amortecimento 2L/R =", 2*L/R, "s")

    plt.figure()
    plt.plot(t, Q)
    plt.ylabel('Q(t)')
    plt.xlabel('tempo (t)')
    plt.xlim(0.0, 125.0)
    plt.title('Carga: Q(t) ')

    plt.figure()
    plt.plot(t, I)
    plt.ylabel('I(t)')
    plt.xlabel('tempo (t)')
    plt.xlim(0.0, 125.0)
    plt.title('Corrente: I(t)')

    plt.figure()
    plt.plot(Q, I)
    plt.ylabel('I(t)')
    plt.xlabel('Q(t)')
    plt.title('Espaço de fases (Q, I)')
    plt.show()


# =============================================================================
# Part II.1 — Triatomic molecule
# =============================================================================

# Condições iniciais [(x1, v1), (x2, v2), (x3, v3)]
CONDICOES_INICIAIS = {
    1: [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],    # everything moves together: translation
    2: [[0.0, 1.0], [0.0, 0.0], [0.0, 1.0]],    # translation + oscillation of the centre
    3: [[0.0, 1.0], [0.0, 0.0], [0.0, -1.0]],   # pure symmetric stretch (x2 = 0)
    4: [[0.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],  # displaced at rest: mixture of modes
}


def exercicio7(condicoes=(1, 2, 3, 4), N=20000):
    """RK4 for m1 = m3 = 1, m2 = 16, k1 = k2 = 1 over 200 s."""
    k1 = k2 = 1.0
    m1, m2, m3 = 1.0, 16.0, 1.0

    def derivadas(t, y):
        x1, x2, x3, v1, v2, v3 = y
        dv1 = (-k1/m1)*(x1-x2)
        dv2 = (-k1/m2)*(x2-x1)-(k2/m2)*(x2-x3)
        dv3 = (-k2/m3)*(x3-x2)
        return np.array([v1, v2, v3, dv1, dv2, dv3])

    t0, tf = 0.0, 200.0
    print("Passo utilizado:", (tf-t0)/N)
    for c in condicoes:
        ci = CONDICOES_INICIAIS[c]
        print("Condição inicial", c, ":", ci)
        y0 = [ci[0][0], ci[1][0], ci[2][0], ci[0][1], ci[1][1], ci[2][1]]
        t, y = rk4(derivadas, y0, t0, tf, N)

        fig, ax = plt.subplots(3, 1, sharex=True)
        for i in range(3):
            ax[i].plot(t, y[:, i])
            ax[i].set_ylabel('X%d(t)' % (i+1))
        ax[2].set_xlabel('t')
        ax[0].set_title('Condição inicial %d' % c)
    plt.show()


# =============================================================================
# Part II.2 — Normal modes (Jacobi method)
# =============================================================================

def off(A):
    """Frobenius norm of the off-diagonal part; the method stops when it is ~0."""
    a = 0.0
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                a += np.square(A[i][j])
    return np.sqrt(a)


def sign(x):
    if x >= 0:
        return 1.0
    return -1.0


def cos_sin(A, j, k):
    """cos and sin of the Jacobi rotation angle that zeroes A[j][k]."""
    if A[j][j] == A[k][k]:
        return np.cos(np.pi/4), np.sin(np.pi/4)
    t = (2*(A[k][j])/(A[k][k] - A[j][j]))
    x = np.sqrt(1/(1 + np.square(t)))
    C = np.sqrt((1+x)/2)
    S = (sign(t))*(np.sqrt((1-x)/2))
    return C, S


def Al(A, j, k, C, S):
    """Applies the rotation to A: A' = Rᵀ A R (only rows/columns j, k change)."""
    B = matrix_append(len(A))
    for r in range(len(A)):
        for s in range(len(A)):
            if (r != j and r != k) and (s != j and s != k):
                B[r][s] = A[r][s]
            if r != j and r != k:
                B[r][j] = (C)*A[r][j] - (S)*A[r][k]
                B[j][r] = B[r][j]
                B[r][k] = (S)*A[r][j] + (C)*A[r][k]
                B[k][r] = B[r][k]

    B[j][j] = (np.square(C))*A[j][j] + (np.square(S))*A[k][k] - 2*((S)*(C))*A[j][k]
    B[j][k] = 0.0
    B[k][j] = 0.0
    B[k][k] = (np.square(S))*A[j][j] + (np.square(C))*A[k][k] + 2*((S)*(C))*A[j][k]
    return B


def Ql(Q, j, k, C, S):
    """Accumulates the rotation in Q (Q ends up holding the eigenvectors as columns)."""
    Ql = [linha[:] for linha in Q]
    for r in range(len(Q)):
        Ql[r][j] = C*Q[r][j] - S*Q[r][k]
        Ql[r][k] = S*Q[r][j] + C*Q[r][k]
    return Ql


def jacobi(A, e=1e-10, Nmax=10000):
    """Cyclic Jacobi method for a symmetric matrix A.

    Each rotation sets one off-diagonal element to zero. Repeating over all
    pairs (j, k) makes A converge to a diagonal matrix of eigenvalues.
    Returns (eigenvalues, eigenvector matrix Q, number of rotations).
    """
    n = len(A)
    indice = [(j, k) for j in range(n) for k in range(j+1, n)]
    Q = matrix_identity(n)
    inter = 0
    while off(A) > e and inter < Nmax:
        j, k = indice[inter % len(indice)]     # cycles through the pairs (j,k), k>j
        if A[j][k] != 0.0:
            C, S = cos_sin(A, j, k)
            A, Q = Al(A, j, k, C, S), Ql(Q, j, k, C, S)
        inter += 1
    l = [A[i][i] for i in range(n)]
    return l, np.array(Q), inter


def mostra_vetores(l, Q):
    """Prints the eigenvectors normalised so the first non-zero component is 1."""
    for j in np.argsort(l):
        v = Q[:, j]
        primeiro = v[np.argmax(np.abs(v) > 1e-8)]
        print("Valor próprio %8.5f -> vector próprio:" % l[j],
              " ".join("%8.5f" % c for c in v/primeiro))


def exercicio9():
    """Normal modes of the triatomic molecule with equal masses (ω0 = 1)."""
    A = [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]]
    l, Q, inter = jacobi(A)
    print("Rotações de Jacobi:", inter)
    print("Valores próprios (ω/ω0)²:", [round(float(x), 10) for x in sorted(l)])
    print("Frequências ω/ω0:", np.sqrt(np.abs(sorted(l))))
    mostra_vetores(l, Q)
    print("(0: translação; 1: estiramento simétrico; 3: estiramento antisimétrico)")


def matriz_cadeia(n):
    """Matrix of a chain of n equal masses and n-1 equal springs (free ends)."""
    M = np.zeros((n, n))
    for i in range(n):
        M[i][i] = 2.0
        if i + 1 < n:
            M[i][i+1] = M[i+1][i] = -1.0
    M[0][0] = M[n-1][n-1] = 1.0
    return M.tolist()


def exercicio10(n=9):
    """Normal modes of a chain of 9 masses; plot of the 4 lowest-frequency modes."""
    A = matriz_cadeia(n)
    l, Q, inter = jacobi(A)
    ordem = np.argsort(l)
    print("Rotações de Jacobi:", inter)
    for i, j in enumerate(ordem):
        print(i+1, "ª", "Valor próprio", ":", "%.6f" % l[j],
              "  (analítico 4sin²(%dπ/18) = %.6f)" % (i, 4*np.sin(i*np.pi/(2*n))**2))

    plt.figure()
    massas = np.arange(1, n+1)
    for j in ordem[:4]:
        plt.plot(massas, Q[:, j], 'o-', label='ω²/ω0² = %.3f' % l[j])
    plt.xlabel('índice da massa i')
    plt.ylabel('deslocamento X_i')
    plt.title('Quatro modos próprios de menor frequência')
    plt.legend()
    plt.show()


EXERCICIOS = {
    '2': exercicio2,
    '4': exercicio4,
    '5': exercicio5,
    '7': exercicio7,
    '9': exercicio9,
    '10': exercicio10,
}


if __name__ == '__main__':
    args = sys.argv[1:]
    ci = None
    if '--ci' in args:
        i = args.index('--ci')
        ci = [int(args[i+1])]
        del args[i:i+2]
    for chave in args or list(EXERCICIOS):
        print("\n" + "=" * 70 + "\nExercício " + chave + "\n" + "=" * 70)
        if chave == '7' and ci:
            exercicio7(ci)
        else:
            EXERCICIOS[chave]()
