# -*- coding: utf-8 -*-
"""
Trabalho 3 (Física Computacional) — Cordas, ondas e transições de fase
=======================================================================

This file joins the old scripts ``exercicio2.py``, ``exercicio5.py`` (and its
copy ``Onda na água.py``), ``Modelo spin tentativa.py``, ``exercício 7.py``,
``exercicio 8.py``, ``exercicio 9.py`` and ``exercicio 10.py`` into one module.
Each exercise is a function, and they can be run separately.

Part I — Vibrating string (exercise 2)
--------------------------------------
Wave equation in 1+1 dimensions, u_tt = c² u_xx, with c = sqrt(T/µ) = 0.5 m/s
(T = 0.25 N, µ = 1 kg/m, L = 1 m). Both ends are fixed, u(0,t) = u(L,t) = 0,
and u_t(x,0) = 0. Separation of variables gives the normal modes

    u_n(x,t) = A sin(nπx/L) cos(nπct/L),   n = 1, 2, 3, ...

The numerical solution uses the explicit centred-difference (leapfrog) scheme

    u_i^{k+1} = r² (u_{i+1}^k + u_{i-1}^k) + 2(1 - r²) u_i^k - u_i^{k-1},
    r = c h_t / h_x

which is stable for r ≤ 1 (CFL condition). The first step uses
u_t(x,0) = 0: u_i^1 = r²/2 (u_{i+1}^0 + u_{i-1}^0) + (1 - r²) u_i^0 + h_t u_t.

Part II — Waves on the surface of water (exercise 5)
----------------------------------------------------
Wave equation in 2+1 dimensions, u_tt = c² (u_xx + u_yy), on a square tank
of side 1 m with c = 0.5 m/s. The water at the edges does not move
(u = 0 on the boundary). A stone dropped at the centre creates the initial
Gaussian bump u(x,y,0) = exp(-500 [(x-½)² + (y-½)²]) with u_t = 0. The bump
splits into a circular wave that travels outwards, reflects at the walls
(with a sign change, since the boundary is fixed) and interferes with itself.
Stability: c h_t sqrt(1/h_x² + 1/h_y²) ≤ 1.

Part III — 2D Ising model with the Metropolis algorithm (exercises 6–10)
-----------------------------------------------------------------------
Spins s_i = ±1 on an Nx × Ny square lattice with FREE boundary conditions
and energy (H = 0)

    E = -J Σ_<ij> s_i s_j        (sum over nearest-neighbour bonds),
    M = Σ_i s_i.

Metropolis: a flip of spin i changes the energy by ΔE = 2 J s_i Σ_nb s_j and
is accepted with probability min(1, exp(-ΔE/k_B T)). This samples the
Boltzmann distribution p ∝ exp(-E/k_B T). Here one Monte Carlo step (sweep)
is a "checkerboard" update: first all the black sites, then all the white
sites. Sites of one colour do not touch each other, so they can all be tried
at once (vectorised with numpy). This is a sequential sweep in a fixed
order, so detailed balance holds.

From the averages (taken after n0 thermalisation steps, J = k_B = 1):

    <E>/N, <|M|>/N,
    C/N = (<E²> - <E>²) / (N T²)      (specific heat),
    χ/N = (<M²> - <|M|>²) / (N T)     (magnetic susceptibility).

The infinite 2D lattice has a phase transition (Onsager) at
T_c = 2/ln(1 + √2) ≈ 2.269: below T_c the system is ferromagnetic (|M| ≈ N),
above it is paramagnetic (M ≈ 0). The exact (Onsager) solution is computed
in ``onsager`` and replaces the old data file 'Isi2DH02.dat'.

* 6: validation on a 3×3 lattice. Full enumeration of the 2⁹ = 512
  configurations gives the degeneracy W(E), Z and p(E). These are compared
  with the energy histogram of a Metropolis run (T = 1 and T = 5).
* 7: thermalisation. Running averages of E and |M| on a 6×6 lattice at T = 2,
  starting from an ordered and from a random configuration.
* 8: temperature sweep T = 0.2 ... 5 on 6×6, compared with the exact solution.
* 9: the same for L = 8, 16 and 40. Finite-size effects: the peak of C grows
  and sharpens, and M drops more abruptly near T_c as L increases.
* 10: typical configurations of a 40×40 lattice below, at, and well above the
  specific-heat peak (big domains -> fractal clusters -> random noise).
  Final configurations are written to the folder 'configuracoes/'.

Usage
-----
    python trabalho3.py              # run every exercise
    python trabalho3.py 2 5          # only the string and the water wave
    python trabalho3.py 8 --ns 2000  # fewer Monte Carlo steps (faster)
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk


# =============================================================================
# Part I — Vibrating string
# =============================================================================

def u_exata(x, t, A, n, L, c):
    """Normal mode n of the fixed-fixed string: A sin(nπx/L) cos(nπct/L)."""
    pi = np.pi
    return A*np.sin((n*pi*x)/L)*np.cos((n*pi*c*t)/L)


def corda(n, L=1.0, c=0.5, A=2.5, tf=1.0, Nx=121, Nt=120, dt_u=0.0):
    """Leapfrog solution of the string for the initial mode n.
    Returns x, t and the matrix u[k, i] (time k, position i)."""
    x = np.linspace(0, L, Nx)
    hx = x[1] - x[0]
    ht = tf/Nt
    r2 = np.square(c*ht/hx)
    assert r2 <= 1.0, "Instável: c*ht/hx tem de ser <= 1"

    u = np.zeros((Nt+1, Nx))
    u[0] = A*np.sin((n*np.pi*x)/L)            # u(x, 0); the ends stay at 0
    # First step, using du/dt(x, 0) = dt_u
    u[1, 1:-1] = 0.5*r2*(u[0, 2:] + u[0, :-2]) + (1.0 - r2)*u[0, 1:-1] + ht*dt_u
    for k in range(1, Nt):
        u[k+1, 1:-1] = r2*(u[k, 2:] + u[k, :-2]) + 2*(1.0 - r2)*u[k, 1:-1] - u[k-1, 1:-1]
    return x, np.linspace(0, tf, Nt+1), u


def exercicio2():
    """Modes n = 1, 2, 3 at 6 times between 0 and 1 s: numerical (solid)
    vs exact (dashed). Mode n has period 2L/(nc) = 4/n s, so in 1 s mode 1
    makes a quarter of an oscillation, mode 2 half and mode 3 three quarters."""
    T, mi, L = 0.25, 1.0, 1.0
    c = np.sqrt(T/mi)
    Nt = 120
    instantes = np.linspace(0, Nt, 6).astype(int)     # 6 time indices
    cores = {1: 'r', 2: 'b', 3: 'k'}

    resultados = {n: corda(n, L, c, Nt=Nt) for n in (1, 2, 3)}
    print("c =", c, "m/s;  r = c*ht/hx =", c*(1.0/Nt)/(resultados[1][0][1]))

    fig, ax = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    fig.suptitle('Corda vibrante: numérico (—) vs exato (--)')
    for p, k in enumerate(instantes):
        a = ax.flat[p]
        for n, (x, t, u) in resultados.items():
            a.plot(x, u[k], cores[n], label='n=%d' % n)
            a.plot(x, u_exata(x, t[k], 2.5, n, L, c), cores[n] + '--')
            erro = np.max(np.abs(u[k] - u_exata(x, t[k], 2.5, n, L, c)))
            if p == len(instantes) - 1:
                print("n = %d: erro máximo em t = %.2f s: %.2e" % (n, t[k], erro))
        a.set_title('t = %.2f s' % t[k])
        a.set_xlabel('x (m)')
        a.set_ylim(-3.0, 3.0)
        a.grid(True)
    ax[0, 0].set_ylabel('Amplitude')
    ax[1, 0].set_ylabel('Amplitude')
    ax[0, 0].legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# Part II — Waves on the surface of water
# =============================================================================

def u0_agua(x, y):
    """Initial perturbation: Gaussian bump at the centre of the tank."""
    return np.exp((-500)*(np.square(x - 0.5) + np.square(y - 0.5)))


def onda_agua(c=0.5, L=1.0, Nx=50, Ny=50, Nt=200, tf=3.0, dt_u=0.0):
    """Explicit scheme for u_tt = c²(u_xx + u_yy) with u = 0 at the edges.
    Returns x, y, t and u[k, i, j]."""
    x = np.linspace(0, L, Nx+1)
    y = np.linspace(0, L, Ny+1)
    hx, hy, ht = L/Nx, L/Ny, tf/Nt
    rx, ry = np.square(c*ht/hx), np.square(c*ht/hy)
    assert rx + ry <= 1.0, "Instável: diminua ht"

    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.zeros((Nt+1, Nx+1, Ny+1))
    u[0] = u0_agua(X, Y)
    u[0, 0, :] = u[0, -1, :] = u[0, :, 0] = u[0, :, -1] = 0.0

    def laplaciano(U):
        return (rx*(U[2:, 1:-1] - 2*U[1:-1, 1:-1] + U[:-2, 1:-1])
                + ry*(U[1:-1, 2:] - 2*U[1:-1, 1:-1] + U[1:-1, :-2]))

    u[1, 1:-1, 1:-1] = 0.5*laplaciano(u[0]) + u[0, 1:-1, 1:-1] + ht*dt_u
    for k in range(1, Nt):
        u[k+1, 1:-1, 1:-1] = laplaciano(u[k]) + 2*u[k, 1:-1, 1:-1] - u[k-1, 1:-1, 1:-1]
    return x, y, np.linspace(0, tf, Nt+1), u


def exercicio5():
    """Surface of the water at 6 times between 0 and 3 s."""
    x, y, t, u = onda_agua()
    X, Y = np.meshgrid(x, y, indexing='ij')
    instantes = np.linspace(0, len(t)-1, 6).astype(int)

    fig = plt.figure(figsize=(12, 8))
    for p, k in enumerate(instantes):
        ax = fig.add_subplot(2, 3, p+1, projection='3d')
        ax.plot_wireframe(X, Y, u[k], rstride=1, cstride=1, color='b', linewidth=0.4)
        ax.set_zlabel('Amplitude')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('t = %.2f s' % t[k])
        ax.set_zlim(-0.5, 1.0)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Part III — Ising model, Metropolis algorithm
# =============================================================================
J = 1.0
NS_PADRAO = 10000     # default number of Monte Carlo steps (can be changed with --ns)


def soma_vizinhos(S):
    """Sum of the 4 nearest neighbours of every site (free boundaries: the
    missing neighbours outside the lattice count as 0)."""
    P = np.pad(S, 1)
    return P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:]


def energia(S, J=J):
    """E = -J Σ_<ij> s_i s_j over horizontal and vertical bonds."""
    return -J*(np.sum(S[1:, :]*S[:-1, :]) + np.sum(S[:, 1:]*S[:, :-1]))


def estado_ordenado(Nx, Ny):
    return np.ones((Nx, Ny), dtype=int)


def estado_aleatorio(Nx, Ny, rng):
    return 2*rng.integers(0, 2, size=(Nx, Ny)) - 1


def mascaras_xadrez(Nx, Ny):
    """Black and white sublattices of the checkerboard."""
    ix, iy = np.indices((Nx, Ny))
    preto = (ix + iy) % 2 == 0
    return preto, ~preto


def passo_mc(S, T, mascaras, rng, J=J):
    """One Monte Carlo step: Metropolis flip attempts on every spin."""
    for m in mascaras:
        dE = 2*J*S*soma_vizinhos(S)                 # energy change if the spin flips
        aceita = m & ((dE <= 0) | (rng.random(S.shape) < np.exp(-np.maximum(dE, 0)/T)))
        S[aceita] *= -1


def metropolis(S, T, ns, n0, rng, J=J, serie=False):
    """Runs ns Monte Carlo steps at temperature T (S is changed in place).

    The averages use only the steps after the first n0 (thermalisation).
    Returns a dict with <E>, <E²>, <|M|>, <M²>, C/N, χ/N and the energy
    histogram. If serie=True it also returns E and |M| at every step.
    """
    N = S.size
    mascaras = mascaras_xadrez(*S.shape)
    Es = np.empty(ns)
    Ms = np.empty(ns)
    for ip in range(ns):
        passo_mc(S, T, mascaras, rng, J)
        Es[ip] = energia(S, J)
        Ms[ip] = np.sum(S)
    E, M = Es[n0:], np.abs(Ms[n0:])
    r = {
        'E': E.mean(), 'E2': np.mean(E**2), 'M': M.mean(), 'M2': np.mean(M**2),
        'E/N': E.mean()/N, 'M/N': M.mean()/N,
        'C/N': (np.mean(E**2) - E.mean()**2)/(N*T**2),
        'X/N': (np.mean(M**2) - M.mean()**2)/(N*T),
        'hist': dict(zip(*np.unique(E, return_counts=True))),
    }
    if serie:
        r['serie_E'], r['serie_M'] = Es, np.abs(Ms)
    return r


def onsager(T, J=J):
    """Exact solution of the infinite 2D Ising model (Onsager, Yang).
    Returns E/N, |M|/N and C/N (C from the numerical derivative dE/dT)."""
    T = np.asarray(T, dtype=float)
    K = J/T
    k = 2*np.sinh(2*K)/np.cosh(2*K)**2
    E = -J/np.tanh(2*K)*(1 + (2/np.pi)*(2*np.tanh(2*K)**2 - 1)*ellipk(k**2))
    Tc = 2*J/np.log(1 + np.sqrt(2))
    M = np.where(T < Tc, np.abs(1 - np.sinh(2*K)**(-4))**(1/8), 0.0)
    C = np.gradient(E, T)
    return E, M, C


def enumeracao(Nx, Ny, J=J):
    """Full enumeration of the 2^(Nx·Ny) configurations: returns {E: W(E)}."""
    N = Nx*Ny
    W = {}
    for conf in range(2**N):
        bits = (conf >> np.arange(N)) & 1
        S = (2*bits - 1).reshape(Nx, Ny)
        E = energia(S, J)
        W[E] = W.get(E, 0) + 1
    return dict(sorted(W.items()))


def exercicio6(ns=11000, n0=1000, seed=1):
    """3×3 lattice: there are 2⁹ = 512 configurations and 12 bonds, so
    -12 J ≤ E ≤ +12 J. The minimum comes from the 2 fully aligned states and
    the maximum from the 2 chessboard (antiferromagnetic) states. The
    Metropolis histogram psim(E) must reproduce p(E) = W(E) e^{-E/T} / Z."""
    rng = np.random.default_rng(seed)
    W = enumeracao(3, 3)
    print("Número de configurações:", sum(W.values()))
    print("  E     W(E)")
    for E, w in W.items():
        print("%4d  %6d" % (E, w))

    for T in (1.0, 5.0):
        Es = np.array(list(W.keys()), dtype=float)
        Ws = np.array(list(W.values()), dtype=float)
        Z = np.sum(Ws*np.exp(-Es/T))
        p = Ws*np.exp(-Es/T)/Z
        S = estado_aleatorio(3, 3, rng)
        r = metropolis(S, T, ns, n0, rng)
        print("\nT = %.1f  ->  Z = %.6g" % (T, Z))
        print("  E      p(E)      psim(E)")
        for E, pe in zip(Es, p):
            psim = r['hist'].get(E, 0)/(ns - n0)
            print("%4d  %.6f  %.6f" % (E, pe, psim))


def exercicio7(ns=NS_PADRAO, T=2.0, L=6, seed=2):
    """Running averages (1/nt) Σ E_i and (1/nt) Σ |M_i| on a 6×6 lattice at
    T = 2, starting from an ordered and from a random configuration. The
    ordered start is already near equilibrium at T = 2 < T_c. The random one
    starts far away and needs some hundreds of steps. This is why the first
    n0 steps are discarded."""
    rng = np.random.default_rng(seed)
    nt = np.arange(1, ns+1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for nome, S in (('Spins Ordenados', estado_ordenado(L, L)),
                    ('Spins Desordenados', estado_aleatorio(L, L, rng))):
        r = metropolis(S, T, ns, 0, rng, serie=True)
        ax[0].plot(nt, np.cumsum(r['serie_E'])/nt, label=nome)
        ax[1].plot(nt, np.cumsum(r['serie_M'])/nt, label=nome)
    ax[0].set_title('Energia')
    ax[0].set_ylabel('(1/nt) Σ E_i')
    ax[1].set_title('Magnetização')
    ax[1].set_ylabel('(1/nt) Σ |M_i|')
    for a in ax:
        a.set_xlabel('nt (passos MC)')
        a.set_xscale('log')
        a.legend()
    plt.tight_layout()
    plt.show()


def varrer_temperatura(L, temperaturas, ns, n0, rng, pasta=None):
    """Temperature sweep starting from the ordered state at low T. The final
    state of each T is the initial state of the next one."""
    S = estado_ordenado(L, L)
    res = []
    for T in temperaturas:
        r = metropolis(S, T, ns, n0, rng)
        res.append(r)
        if pasta:
            np.savetxt(os.path.join(pasta, 'L%d_T%.1f.dat' % (L, T)), S, fmt='%d')
    return {q: np.array([r[q] for r in res]) for q in ('E/N', 'M/N', 'C/N', 'X/N')}


def grafico_comparacao(temperaturas, resultados):
    """<E>/N, <|M|>/N, C/N and χ/N vs T, with the exact solution for L = ∞."""
    Tex = np.linspace(0.2, 5.0, 2000)
    Eex, Mex, Cex = onsager(Tex)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for (q, titulo), a, ex in zip((('E/N', 'Energia <E>/N'), ('M/N', 'Magnetização <|M|>/N'),
                                   ('C/N', 'Calor específico C/N'), ('X/N', 'Susceptibilidade χ/N')),
                                  ax.flat, (Eex, Mex, Cex, None)):
        if ex is not None:
            a.plot(Tex, ex, 'k', label='Exato (L = ∞)')
        for L, r in resultados.items():
            a.plot(temperaturas, r[q], 'o--', ms=4, label='L = %d' % L)
        a.set_title(titulo)
        a.set_xlabel('T (J/kB)')
        a.legend()
    plt.tight_layout()
    plt.show()


def exercicio8(ns=NS_PADRAO, n0=1000, seed=3):
    """6×6 lattice, T = i·0.2, i = 1..25."""
    rng = np.random.default_rng(seed)
    temperaturas = 0.2*np.arange(1, 26)
    r = varrer_temperatura(6, temperaturas, ns, n0, rng)
    print("   T     <E>/N    <|M|>/N    C/N      χ/N")
    for i, T in enumerate(temperaturas):
        print("%5.1f  %8.4f  %8.4f  %7.4f  %7.4f" % (T, r['E/N'][i], r['M/N'][i], r['C/N'][i], r['X/N'][i]))
    grafico_comparacao(temperaturas, {6: r})


def exercicio9(ns=NS_PADRAO, n0=None, seed=4, tamanhos=(8, 16, 40)):
    """L = 8, 16, 40: the curves approach the exact solution as L grows (the
    peak of C/N moves towards T_c ≈ 2.269 and gets higher)."""
    rng = np.random.default_rng(seed)
    n0 = ns//2 if n0 is None else n0
    temperaturas = 0.2*np.arange(1, 26)
    resultados = {}
    for L in tamanhos:
        print("A simular L =", L, "...")
        resultados[L] = varrer_temperatura(L, temperaturas, ns, n0, rng)
        i = np.argmax(resultados[L]['C/N'])
        print("   pico de C/N em T = %.1f (C/N = %.3f)" % (temperaturas[i], resultados[L]['C/N'][i]))
    grafico_comparacao(temperaturas, resultados)


def exercicio10(ns=NS_PADRAO, n0=1000, seed=5, L=40):
    """Final 40×40 configurations: below the peak of C (large ordered domains),
    at the peak (clusters of every size, critical point) and at T = 5
    (disordered, short-range correlations). One file per temperature is
    written to 'configuracoes/'."""
    rng = np.random.default_rng(seed)
    pasta = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configuracoes')
    os.makedirs(pasta, exist_ok=True)
    temperaturas = 0.2*np.arange(1, 26)
    r = varrer_temperatura(L, temperaturas, ns, n0, rng, pasta=pasta)
    Tpico = temperaturas[np.argmax(r['C/N'])]
    print("Configurações finais escritas em", pasta)
    print("Pico do calor específico em T = %.1f" % Tpico)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, T in zip(ax, (1.6, Tpico, 5.0)):
        S = np.loadtxt(os.path.join(pasta, 'L%d_T%.1f.dat' % (L, T)))
        a.imshow(S, cmap='gray', vmin=-1, vmax=1)
        a.set_title('T = %.1f J/kB' % T)
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.show()


EXERCICIOS = {
    '2': exercicio2,
    '5': exercicio5,
    '6': exercicio6,
    '7': exercicio7,
    '8': exercicio8,
    '9': exercicio9,
    '10': exercicio10,
}


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--ns' in args:
        i = args.index('--ns')
        NS_PADRAO = int(args[i+1])
        del args[i:i+2]
        # the functions captured the default at definition time, so pass it explicitly
        for k in ('7', '8', '9', '10'):
            f = EXERCICIOS[k]
            EXERCICIOS[k] = (lambda f: lambda: f(ns=NS_PADRAO))(f)
    for chave in args or list(EXERCICIOS):
        print("\n" + "=" * 70 + "\nExercício " + chave + "\n" + "=" * 70)
        EXERCICIOS[chave]()
