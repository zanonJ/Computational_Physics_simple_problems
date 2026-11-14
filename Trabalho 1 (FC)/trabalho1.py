# -*- coding: utf-8 -*-
"""
Trabalho 1 (Física Computacional) — Poço quadrado finito e momento quadrupolar
==============================================================================

This file joins the old scripts ``exercío3.py``, ``exercício 4.py``,
``exercício 5.py``, ``exercício 9.py`` and ``exercício 9-3.py`` into one
module. Each exercise is a function, and they can be run separately.

Part I — Finite square well (exercises 3, 4, 5)
-----------------------------------------------
A particle of mass m is in the potential V(x) = V0 for |x| > a and V(x) = 0
for |x| < a. For bound states (E < V0) the wave function is oscillatory
inside the well (k2 = sqrt(2mE)/hbar) and decays exponentially outside
(k1 = sqrt(2m(V0-E))/hbar). Requiring Psi and dPsi/dx to be continuous at
x = ±a gives two transcendental equations:

    even parity (Psi(-x) =  Psi(x)):  k1 =  k2 tan(k2 a)
    odd  parity (Psi(-x) = -Psi(x)):  k1 = -k2 cot(k2 a)

With the dimensionless variables  alpha = k2 a = sqrt(2ma²E)/hbar  and
p = sqrt(2ma²V0)/hbar, they become

    f1(alpha) = alpha sin(alpha) - sqrt(p² - alpha²) cos(alpha) = 0   (even)
    f2(alpha) = alpha cos(alpha) + sqrt(p² - alpha²) sin(alpha) = 0   (odd)

and the energy, in units of hbar²/(2ma²), is simply E = alpha².

* Exercise 3: graphical solution for V0 = 36 (p = 6). The intersections of
  sqrt(p² - alpha²) with alpha·tan(alpha) and -alpha·cot(alpha) are the bound
  states.
* Exercise 4: numerical solution with four root-finding methods: bisection,
  regula falsi (false position), Newton-Raphson and secant. Stopping
  criteria: |f(alpha)| <= 1e-6 and |Δalpha| <= 1e-6.
* Exercise 5: ground-state energy E0 as a function of V0 (10 ... 1e6). As
  V0 -> infinity, E0 tends to the infinite-well value (pi/2)² ≈ 2.4674.
* Exercise 6: Newton interpolating polynomial (divided differences) of E0(V0)
  through V0 = 1, 5, 20, 40, 60, 80, evaluated at V0 = 36 and compared with
  the value from exercise 4.

Part II — Quadrupole moment of the hydrogen atom (exercise 9)
-------------------------------------------------------------
For an axially symmetric charge density rho(r) = f(r) g(cos θ), the
quadrupole tensor factorises as Q_ij = F · A_ij with

    F   = ∫_0^∞ r⁴ f(r) dr                       (radial part)
    A33 = 2π ∫_-1^1 g(η) (3η² - 1) dη,  η = cos θ  (angular part)

For hydrogen states with m = 0, rho = |psi_nl0|² = f_nl(r) g_l(η). The sign
of Q33 tells the shape of the cloud: Q33 > 0 is prolate ("cigar") and
Q33 < 0 is oblate ("pancake").

* 9.1: checks that ∫ f r² dr = 1 and 2π ∫ g dη = 1 (normalisation).
* 9.2: plots the radial densities and the angular part in the x–z plane.
* 9.3: computes F, A33 and Q33 = F·A33.

All integrals use Gauss-Legendre quadrature. For [0, ∞[ the change of
variable r = tan(π/4 (1 + x)), with dr = (π/4) dx / cos²(π/4 (1 + x)), maps
x ∈ [-1, 1] onto r ∈ [0, ∞[.

Usage
-----
    python trabalho1.py            # run every exercise
    python trabalho1.py 4 5        # run only exercises 4 and 5
    python trabalho1.py 9.3        # run only exercise 9.3
"""
import sys
import math as mat

import numpy as np
import pandas
import matplotlib.pyplot as plt


# =============================================================================
# Part I — Finite square well
# =============================================================================

# p: square root of the potential V0 (in units of hbar²/2ma²); V0 = 36 -> p = 6
p = 6.0


def f1(x):
    """Even-parity condition: f1(alpha) = alpha sin(alpha) - sqrt(p²-alpha²) cos(alpha)."""
    f1 = x*np.sin(x) - (np.sqrt(p**2 - x**2))*np.cos(x)
    return(f1)


def f2(x):
    """Odd-parity condition: f2(alpha) = alpha cos(alpha) + sqrt(p²-alpha²) sin(alpha)."""
    f2 = x*np.cos(x) + (np.sqrt(p**2 - x**2))*np.sin(x)
    return (f2)


def df1(x):
    """Analytical derivative of f1 (used by Newton-Raphson)."""
    df1 = (1.0 + np.sqrt(p**2 - x**2))*(np.sin(x)) + (1.0 + (1/(np.sqrt(p**2 - x**2))))*x*np.cos(x)
    return(df1)


def df2(x):
    """Analytical derivative of f2 (used by Newton-Raphson)."""
    df2 = (1.0 + np.sqrt(p**2 - x**2))*(np.cos(x)) - (1.0 + (1/(np.sqrt(p**2 - x**2))))*x*np.sin(x)
    return(df2)


# --- Root-finding methods ----------------------------------------------------
# Every method returns (alpha, E = alpha², |f(alpha)|, |Δalpha|, iterations),
# or None if it did not converge within Nmax iterations.

def Bi(a, b, d, Nmax, f):
    """Bisection: halves the bracket [a, b] keeping the sign change of f."""
    inter = 1
    c = (a+b)/(2)
    while (abs(f(c)) > d or abs(a - b) > d) and (inter < Nmax):
        if (f(a)*f(c)) < 0:
            b = c
        else:
            a = c

        c = (a+b)/(2)
        inter = inter + 1

    if inter == Nmax:
        print("não deu certo")
        return None
    else:
        return(c, c**2, abs(f(c)), abs(a - b), inter)


def FP(a, b, d, Nmax, f):
    """Regula falsi: like bisection, but the new point is where the chord
    between (a, f(a)) and (b, f(b)) crosses zero."""
    inter = 1
    c = ((a*f(b)) - (b*f(a)))/(f(b)-f(a))
    while (abs(f(c)) > d) or (abs(a-b) > d) and inter < Nmax:

        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
        c = ((a*f(b) - (b*f(a))))/(f(b)-f(a))
        inter = inter + 1

    if inter == Nmax:
        print("não deu certo")
        return None
    else:
        return(c, c**2, abs(f(c)), abs(b-a), inter)


def NP(w, d, Nmax, f, df):
    """Newton-Raphson: x_{n+1} = x_n - f(x_n)/f'(x_n), starting from w."""
    inter = 1
    c = w
    e = c-w
    while ((abs(f(c)) > d) or (abs(e) > d)) and (inter < Nmax):

        e = - f(c)/(df(c))      # Newton step Δalpha (was never updated before)
        c = c + e
        inter = inter + 1

    if inter == Nmax:
        print("não deu certo")
        return None
    else:
        return(c, c**2, abs(f(c)), abs(e), inter)


def MS(a, b, d, Nmax, f):
    """Secant: Newton-Raphson with the derivative replaced by the slope of the
    line through the last two points (no derivative needed)."""
    inter = 1
    c = (a*f(b) - b*f(a))/(f(b)-f(a))

    while ((abs(f(c)) > d) or (abs(a-b) > d)) and (inter < Nmax):
        a = b
        b = c
        c = (a*f(b) - b*f(a))/(f(b)-f(a))
        inter = inter + 1

    if inter == Nmax:
        print("não deu certo")
        return None
    else:
        return(c, c**2, abs(f(c)), abs(a-b), inter)


def exercicio3():
    """Graphical solution: the intersections of sqrt(36 - alpha²) with
    alpha·tan(alpha) (even) and -alpha·cot(alpha) (odd) are the bound states.
    There are 4 positive roots (2 even + 2 odd); the negative alphas are the
    same states mirrored, since E = alpha²."""
    x = np.linspace(-6, 6, num=1000)
    g1 = np.sqrt(36 - np.square(x))
    g2 = x*np.tan(x)                  # Estados com paridade Postivia
    g3 = -x*(1/(np.tan(x)))           # Estados com paridade Negativa

    # Hide the vertical jumps of tan/cot at their asymptotes
    threshold = 100
    g2[g2 > threshold] = np.inf
    g2[g2 < -threshold] = np.inf
    g3[g3 > threshold] = np.inf
    g3[g3 < -threshold] = np.inf

    print("Gráficos para as Paridades Positiva e Negativa")
    plt.figure()
    plt.plot(x, g1, x, g2)  # Paridade Positiva
    plt.xlabel(r'$\alpha$')
    plt.ylabel('y(x)')
    plt.title('Paridade Positiva')
    plt.ylim(-7, 7)

    plt.figure()
    plt.plot(x, g1, x, g3)  # Paridade Negativa
    plt.xlabel(r'$\alpha$')
    plt.ylabel('y(x)')
    plt.title('Paridade Negativa')
    plt.ylim(-7, 7)
    plt.show()


def exercicio4():
    """Bound-state energies for V0 = 36 with the four root-finding methods.
    The ground state is the even state with the smallest |alpha| (E ≈ 1.81)."""
    tol, Nmax = 10**(-6), 2000
    dados = ["Alfa", "Energy", "|f(alpha)|", "|Alfa|", "  Interações"]
    paridade = ["Positiva", "Positiva", "Positiva", "Positiva"]
    paridade1 = ["Negativa", "Negativa", "Negativa", "Negativa"]

    # Bisection: brackets [-6,-3], [-3,0], [0,3], [3,6]
    u = np.linspace(-6, 6, num=5)
    B = []
    B1 = []
    for i in range(0, len(u)-1):
        B.append(Bi(u[i], u[i+1], tol, Nmax, f1))
        B1.append(Bi(u[i], u[i+1], tol, Nmax, f2))

    print("                   ")
    print("Método: Bissecção")
    print(pandas.DataFrame(B, paridade, dados))
    print(pandas.DataFrame(B1, paridade1, dados))

    # Regula falsi (the odd roots need narrower brackets that skip alpha = 0)
    A = []
    for i in range(0, len(u)-1):
        A.append(FP(u[i+1], u[i], tol, Nmax, f1))

    t1 = np.linspace(-6, 6, num=7)
    A1 = []
    for i in range(0, len(t1)-1):
        if (t1[i] != 0.0) and (t1[i+1] != 0.0):
            A1.append(FP(t1[i], t1[i+1], tol, Nmax, f2))

    print("                   ")
    print("Método: Falsa Posição")
    print(pandas.DataFrame(A, paridade, dados))
    print(pandas.DataFrame(A1, paridade1, dados))

    # Newton-Raphson: one initial guess per root
    t = np.linspace(-4, 4, num=5)
    C = []
    for i in range(0, len(t)):
        if t[i] != 0.0:
            C.append(NP(t[i], tol, Nmax, f1, df1))

    t2 = [-5.0, -3.0, 3.0, 5.0]
    C1 = []
    for i in range(0, len(t2)):
        C1.append(NP(t2[i], tol, Nmax, f2, df2))

    print("                   ")
    print("Newton - Raphson")
    print(pandas.DataFrame(C, paridade, dados))
    print(pandas.DataFrame(C1, paridade1, dados))

    # Secant
    D = []
    for i in range(0, len(t)-1):
        D.append(MS(t[i+1], t[i], tol, Nmax, f1))

    D1 = []
    for i in range(0, len(t1)-1):
        if (t1[i] != 0.0) and (t1[i+1] != 0.0):
            D1.append(MS(t1[i], t1[i+1], tol, Nmax, f2))

    print("                   ")
    print("Método da Secante")
    print(pandas.DataFrame(D, paridade, dados))
    print(pandas.DataFrame(D1, paridade1, dados))


def exercicio5():
    """Ground-state energy E0(V0) with the secant method. The ground state is
    always the first even-parity root, in ]0, pi/2[. When V0 grows the wave
    function barely leaks out of the well and E0 -> (pi/2)² ≈ 2.4674, the
    infinite-well ground state."""
    v0 = [10, 100, 1000, 10000, 100000, 1000000]

    def f(x, v0):
        # Same as f1, but with p² = V0 given explicitly
        f = x*np.sin(x) - (np.sqrt(v0 - x**2))*np.cos(x)
        return(f)

    def MS_v0(a, b, d, Nmax, f, v0):
        inter = 1
        c = (a*f(b, v0) - b*f(a, v0))/(f(b, v0)-f(a, v0))

        while ((abs(f(c, v0)) > d) or (abs(a-b) > d)) and (inter < Nmax):
            a = b
            b = c
            c = (a*f(b, v0) - b*f(a, v0))/(f(b, v0)-f(a, v0))
            inter = inter + 1

        if inter == Nmax:
            print("não deu certo")
            return None
        else:
            return(v0, c, c**2, abs(f(c, v0)), abs(a-b), inter)

    dados = ["V0", "Alfa", "Energy", "|f(alpha)|", "|Alfa|", "  Interações"]
    none = [" ", " ", " ", " ", " ", "  "]

    D = []
    for i in range(0, len(v0)):
        D.append(MS_v0(0.0, 2.0, 10**(-6), 2000, f, v0[i]))

    print("                   ")
    print("Método da Secante")
    print(pandas.DataFrame(D, none, dados))

    E0 = [linha[2] for linha in D]
    plt.figure()
    plt.semilogx(v0, E0, 'o-', label='E0(V0)')
    plt.axhline((mat.pi/2)**2, color='gray', ls='--', label=r'poço infinito $(\pi/2)^2$')
    plt.ylabel('E0 (V0)')
    plt.xlabel('V0')
    plt.legend()
    plt.show()


def exercicio6():
    """Interpolating polynomial of E0(V0). The ground state lies in
    ]0, min(sqrt(V0), pi/2)], where f changes sign, so bisection always works.
    A degree-5 polynomial through widely spaced points only roughly
    reproduces E0(36) = 1.808 (P(36) ≈ 1.76): E0(V0) is not polynomial-like
    (it grows like sqrt(V0) at small V0 and saturates at large V0)."""
    global p                                 # f1 uses the global p = sqrt(V0)
    v0 = [1, 5, 20, 40, 60, 80]

    E0 = []
    for V in v0:
        p_antigo, p = p, np.sqrt(V)
        raiz = Bi(0.0, min(p, mat.pi/2), 10**(-10), 2000, f1)
        p = p_antigo
        E0.append(raiz[1])
    x = np.array(v0, dtype=float)
    g = len(x)

    # Tabela de diferenças divididas d[i][j] = f[x_i, ..., x_{i+j}]
    d = np.zeros((g, g))
    d[:, 0] = E0
    for j in range(1, g):
        for i in range(0, g-j):
            d[i][j] = (d[i+1][j-1] - d[i][j-1])/(x[i+j] - x[i])
    print("Tabela de diferenças divididas:")
    print(pandas.DataFrame(d, ["V0=%g" % v for v in v0], ["ordem %d" % j for j in range(g)]))
    print("Coeficientes do polinómio:", d[0])

    def P(t):   # forma de Newton, avaliada pelo esquema de Horner
        r = d[0][g-1]
        for k in range(g-2, -1, -1):
            r = r*(t - x[k]) + d[0][k]
        return r

    print("P(36) = %.6f   vs   E0(36) = 1.808355 (exercício 4)" % P(36.0))
    t = np.linspace(1, 80, 400)
    plt.figure()
    plt.plot(t, P(t), 'b-', label='Polinómio interpolador')
    plt.plot(x, E0, 'ro', label='E0(V0)')
    plt.plot(36, 1.808355, 'k*', ms=12, label='E0(36), exercício 4')
    plt.xlabel('V0')
    plt.ylabel('E0 (V0)')
    plt.legend()
    plt.show()


# =============================================================================
# Part II — Quadrupole moment of the hydrogen atom
# =============================================================================
# Notation:
# eta = w = cos(theta)
# n = 2,3 e l = 1,2 são números quânticos da tabela apresentado no PDF
# g(n,l,w) returns 2*pi*g(eta) (so that its integral over [-1,1] should be 1)
# f(n,l,r) returns f(r)*r² (so that its integral over [0, inf[ should be 1)
# a0 é o raio de Bohr (0.53 Å)

NPONTOS = 100   # number of Gauss-Legendre points (the old .dat files had 20 or 100)


def pontos_gauss(N=NPONTOS):
    """Nodes x_i and weights w_i of N-point Gauss-Legendre quadrature on [-1, 1].
    Replaces the old 'gauss_N_pontos.dat' files."""
    return np.polynomial.legendre.leggauss(N)


def g(n, l, w):
    """Angular density 2π·g_l(η) for the states of the table."""
    pi = mat.pi
    if l == 1 and (n == 2 or n == 3):
        g1 = (2*pi)*(3/(4*pi))*(w**2)
        return (g1)

    elif l == 2 and n == 3:
        g3 = (2*pi)*(5/(16*pi))*(np.square(3*(w**2) - 1))
        return (g3)


def f(n, l, r, a0=0.53):
    """Radial density times r²: f_nl(r)·r²  (a0 in Å)."""
    if n == 2 and l == 1:
        f1 = (r**2)*((1/(r**2))*(1/(24*a0))*((r/a0)**4)*(np.exp((-r)/a0)))
        return (f1)

    elif n == 3 and l == 1:
        f2 = (r**2)*((1/(r**2))*(1/(216*a0))*(((2*r)/(3*a0))**4)*(np.square(4 - (2*r)/(3*a0)))*(np.exp((-2*r)/(3*a0))))
        return (f2)

    elif n == 3 and l == 2:
        f3 = (r**2)*((1/(r**2))*(1/(1080*a0))*(((2*r)/(3*a0))**6)*(np.exp((-2*r)/(3*a0))))
        return (f3)


def T(x):
    """Change of variable r = tan(π/4·(1+x)), maps [-1, 1] to [0, ∞[."""
    pi = mat.pi
    t = pi/4
    T = np.tan(t*(1+x))
    return(T)


def C(x):
    """cos²(π/4·(1+x)): Jacobian factor of the change of variable, dr = (π/4)/C(x) dx."""
    pi = mat.pi
    t = pi/4
    C = np.square(np.cos(t*(1+x)))
    return C


def GL(func, N=NPONTOS):
    """Gauss-Legendre quadrature of func(x) on [-1, 1]."""
    x, w = pontos_gauss(N)
    return np.sum(func(x)*w)


def GL_inf(func, N=NPONTOS):
    """Gauss-Legendre quadrature of func(r) on [0, ∞[ with r = T(x)."""
    x, w = pontos_gauss(N)
    return np.sum(func(T(x))*(mat.pi/4)*w/C(x))


def exercicio9_1():
    """Checks that the radial and angular densities are each normalised to 1."""
    print("Exercício 9.1")
    print("Normalizações usando o método de Gauss-Legendre com", NPONTOS, "pontos")
    for n, l in [(2, 1), (3, 1), (3, 2)]:
        print("g(%d,%d,w):" % (n, l), GL(lambda w: g(n, l, w)))
    for n, l in [(2, 1), (3, 1), (3, 2)]:
        print("f(%d,%d,r):" % (n, l), GL_inf(lambda r: f(n, l, r)))


def exercicio9_2():
    """Plots of the radial part a0·f(r)·r² vs r/a0, and of the angular part as
    the polar curve (sin θ·g(cos θ), cos θ·g(cos θ)) in the x–z plane (the 3D
    shape is this curve rotated around z). l = 1 is lobed along z (p_z
    orbital); l = 2 has the d_z² shape (two lobes plus a ring)."""
    print("             ")
    print("Exercício 9.2")

    # Parte Radial - f(r)
    r = np.linspace(0.05, 10, num=1000)
    a0 = (0.53)    # Raio de Bohr em Angstron

    B1 = a0*f(2, 1, r/a0)
    B2 = a0*f(3, 1, r/a0)
    B3 = a0*f(3, 2, r/a0)

    plt.figure()
    plt.plot(r/a0, B1, 'r-', label='f1(2,1,r)')
    plt.plot(r/a0, B2, 'b-', label='f2(3,1,r)')
    plt.plot(r/a0, B3, 'black', label='f3(3,2,r)')
    plt.legend()
    plt.title('Parte Radial')
    plt.ylabel('a0*f(r)')
    plt.xlabel('r/a0')
    plt.xlim(0, 14.5)

    # Parte angular g(eta)
    def G(n, l, w):    # Função para mudar a variável eta para cos(teta); teta = w
        pi = mat.pi
        if l == 1 and (n == 2 or n == 3):
            G = (np.sqrt(1/(2*pi)))*(np.sqrt(g(n, l, np.square(np.cos(w)))))
            return (G)
        elif l == 2 and n == 3:
            G = (np.sqrt(1/(2*pi)))*(np.cos(w))*(np.sqrt(g(n, l, np.cos(w))))
            return (G)

    pi = mat.pi
    w = np.linspace(-pi, pi, num=100)

    # Parte positiva (z > 0) e negativa (z < 0)
    C1 = np.sin(w)*G(2, 1, w)        # l = 1
    C2 = np.cos(w)*G(3, 2, w)        # l = 2
    C3 = np.sin(w)*(1)*G(2, 1, w)
    C4 = np.cos(w)*(-1)*G(3, 2, w)

    plt.figure()
    plt.plot(C1, C2, 'blue')
    plt.plot(C3, C4, 'red')
    plt.xlabel('x = sin(teta)*g(cos(teta))')
    plt.ylabel('z = cos(teta)*g(cos(teta))')
    plt.title('Parte Angular')
    plt.show()


def exercicio9_3():
    """Quadrupole moment Q33 = F·A33 (here with a0 = 1, i.e. r in Bohr radii).
    Expected: F = <r²> (30, 180 and 126 for 2p, 3p and 3d), A33 = 4/5 for
    l = 1 and 4/7 for l = 2. All Q33 > 0, so the clouds are prolate
    ("cigar" shaped) along z."""

    def gg(n, l, w):   # integrand of A33: 2π g(η)(3η² - 1)
        return g(n, l, w)*((3*(np.square(w))) - 1)

    def ff(n, l, r):   # integrand of F: r⁴ f(r) = r² · [f(r) r²]
        return (r**2)*f(n, l, r, a0=1.0)

    def Q33(n, l):
        F = GL_inf(lambda r: ff(n, l, r))
        A33 = GL(lambda w: gg(n, l, w))
        return (float(F*A33), float(F), float(A33))

    print("Valores obtidos para o momento quadrupolar Q33 = F*A33 (a0 = 1):")
    print("n=2, l=1:", "(Q33, F, A33) =", Q33(2, 1))
    print("n=3, l=1:", "(Q33, F, A33) =", Q33(3, 1))
    print("n=3, l=2:", "(Q33, F, A33) =", Q33(3, 2))


EXERCICIOS = {
    '3': exercicio3,
    '4': exercicio4,
    '5': exercicio5,
    '6': exercicio6,
    '9.1': exercicio9_1,
    '9.2': exercicio9_2,
    '9.3': exercicio9_3,
}


if __name__ == '__main__':
    escolha = sys.argv[1:] or list(EXERCICIOS)
    for chave in escolha:
        # '9' runs 9.1, 9.2 and 9.3
        for k in [k for k in EXERCICIOS if k == chave or k.startswith(chave + '.')]:
            print("\n" + "=" * 70 + "\nExercício " + k + "\n" + "=" * 70)
            EXERCICIOS[k]()
