# -*- coding: utf-8 -*-
"""
Trabalho 4 (Física Computacional) — Dinâmica Molecular e Difusão Monte Carlo
=============================================================================

This file joins the old scripts ``Exercicio 1.py``, ``Exercicio 2.py``,
``Exercicio 3a.py``, ``Exercicio 3b.py``, ``Exercicio 4.py``,
``Exercicio 5a.py``, ``Exercicio 5b.py`` and ``Exercicio 6.py`` into one
module. Each exercise is a function, and they can be run separately.

Part I — Molecular dynamics of a 2D Lennard-Jones gas (exercises 1–5)
--------------------------------------------------------------------
N particles (mass 1) in a periodic Lx × Ly box interact with the
Lennard-Jones potential, in reduced units (ε = σ = m = k_B = 1):

    u(r) = 4 (r⁻¹² - r⁻⁶),     F(r) = 24/r⁸ (2/r⁶ - 1) · r_vec

The r⁻¹² term is the short-range Pauli repulsion and the -r⁻⁶ term is the
van der Waals attraction. The minimum is at r = 2^(1/6) ≈ 1.12 with
u = -1. Periodic boundary conditions use the minimum-image convention: each
pair interacts through the closest periodic copy.

The equations of motion are integrated with velocity Verlet:

    v(t + dt/2) = v(t) + a(t) dt/2
    r(t + dt)   = r(t) + v(t + dt/2) dt
    v(t + dt)   = v(t + dt/2) + a(t + dt) dt/2

This scheme is time-reversible and symplectic, so the energy error stays
bounded (it oscillates, ~dt²) instead of drifting.

Measured quantities (d = 2, V = Lx·Ly):

    T = 2 <E_kin> / (d N)                         (equipartition)
    P = 2 <E_kin> / (d V) + <φ> / (d V),  φ = Σ_{i<j} r_ij · F_ij   (virial)

* 1: trajectories of particles 5, 6, 7 and 10 (free gas vs interacting gas).
* 2: time reversal: run Nt steps, flip v -> -v, run Nt more. Do we come back?
  Yes for Nt = 1000, but less accurately for Nt = 2000: the dynamics is
  chaotic, so round-off errors grow exponentially (Lyapunov instability).
* 3a: relative energy error η(t) = (E(t) - E(0))/E(0) for dt = 0.01 and
  dt = 0.001 (the error drops ~100×, as expected for O(dt²)).
* 3b: total momentum (v_cm = Σ v_i) is conserved to machine precision,
  because the pair forces obey Newton's 3rd law. init.dat has v_cm ≠ 0, so
  v_cm/N is subtracted from every particle from then on.
* 4: running time averages of E_kin/N, E_pot/N and E/N. Temperature and
  pressure, compared with the ideal gas P = ρT, ρ = N/V.
* 5a/5b: N = 36 particles on a square grid with |v| = v0 (random direction).
  With v0 = 0 the particles stay near their lattice sites (a 2D solid:
  the grid spacing is 1, close to the LJ minimum). With v0 = 2 the
  lattice melts and the particles diffuse like a fluid.

Part II — Diffusion Monte Carlo, 1D harmonic oscillator (exercises 6–10)
-----------------------------------------------------------------------
In imaginary time τ = it the Schrödinger equation becomes a diffusion
equation with a source/sink term:

    ∂ψ/∂τ = ½ ∂²ψ/∂x² - (V(x) - E_R) ψ

A population of "walkers" represents ψ(x, τ). At each step every walker
diffuses, x -> x + η sqrt(Δτ) with η ~ N(0,1), and then is copied
int(q + r) times, with q = exp(-(V(x) - E_R) Δτ) and r uniform in [0,1[
(branching: walkers in low-potential regions multiply, the others die). The
reference energy is adjusted to keep the population near N_P:
E_R -> E_R + α ln(N_P/N). As τ -> ∞ only the ground state survives, so
<E_R> -> E0 and the walker histogram -> φ0(x).
For V = x²/2 (ħ = m = 1): E0 = 1/2, φ0(x) = π^(-1/4) exp(-x²/2).

* 6: initial uniform distribution of walkers in [-1, 1].
* 7: running average <E(τ)> = (1/k) Σ E_R.
* 8: final energy with its error, and the normalised wave function vs the exact one.
* 9: wave function after the first step vs the last step.
* 10: the same with all walkers starting at x = 0.

Usage
-----
    python trabalho4.py              # run every exercise
    python trabalho4.py 1 2 3a       # only some of them
    python trabalho4.py dmc          # only the Diffusion Monte Carlo (6-10)
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

PASTA = os.path.dirname(os.path.abspath(__file__))
INIT = os.path.join(PASTA, 'init.dat')     # initial state: x, y, vx, vy per line


# =============================================================================
# Part I — Molecular dynamics
# =============================================================================

def ler_estado(ficheiro=INIT):
    """Reads x, y, vx, vy (one particle per line)."""
    d = np.loadtxt(ficheiro)
    return d[:, 0].copy(), d[:, 1].copy(), d[:, 2].copy(), d[:, 3].copy()


def escrever_estado(ficheiro, x, y, vx, vy):
    np.savetxt(ficheiro, np.column_stack([x, y, vx, vy]), fmt='%.10f')


def accelALL(x, y, Lx, Ly):
    """Lennard-Jones accelerations (m = 1), total potential energy and virial
    φ = Σ_{i<j} r_ij·F_ij, using the minimum-image convention."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dx -= Lx*np.round(dx/Lx)          # minimum image
    dy -= Ly*np.round(dy/Ly)
    r2 = dx**2 + dy**2
    np.fill_diagonal(r2, np.inf)      # no self-interaction
    ir6 = 1.0/r2**3
    f_r = 24.0*ir6*(2.0*ir6 - 1.0)/r2   # |F|/r
    ax = np.sum(f_r*dx, axis=1)
    ay = np.sum(f_r*dy, axis=1)
    epot = 0.5*np.sum(4.0*(ir6**2 - ir6))   # 1/2: every pair is counted twice
    virial = 0.5*np.sum(f_r*np.where(np.isfinite(r2), r2, 0.0))   # r_ij·F_ij = (|F|/r) r²
    return ax, ay, epot, virial


def vcm(vx, vy):
    """Centre-of-mass 'velocity' Σ v_i (= total momentum in reduced units)."""
    return np.sum(vx), np.sum(vy)


def corrige_vcm(vx, vy):
    """Removes the motion of the centre of mass: v_i -> v_i - v_cm/N."""
    N = len(vx)
    px, py = vcm(vx, vy)
    return vx - px/N, vy - py/N


def dinamica(x, y, vx, vy, Lx, Ly, dt, Nt, Inter=True):
    """Velocity Verlet for Nt steps.

    Returns a dict with the trajectories X[it, i], Y[it, i] (positions kept
    inside the box), the final velocities, and the time series of E_kin,
    E_pot, virial and v_cm. Index 0 is the initial state.
    """
    x, y, vx, vy = (np.array(a, dtype=float) for a in (x, y, vx, vy))
    N = len(x)
    if Inter:
        ax, ay, epot, virial = accelALL(x, y, Lx, Ly)
    else:
        ax, ay, epot, virial = np.zeros(N), np.zeros(N), 0.0, 0.0

    X = np.empty((Nt+1, N))
    Y = np.empty((Nt+1, N))
    ekin = np.empty(Nt+1)
    Epot = np.empty(Nt+1)
    Vir = np.empty(Nt+1)
    VCM = np.empty((Nt+1, 2))

    def guarda(it):
        X[it], Y[it] = x, y
        ekin[it] = 0.5*np.sum(vx**2 + vy**2)
        Epot[it], Vir[it] = epot, virial
        VCM[it] = vcm(vx, vy)

    guarda(0)
    for it in range(1, Nt+1):
        vx += ax*dt/2
        vy += ay*dt/2
        x = np.mod(x + vx*dt, Lx)          # periodic boundary conditions
        y = np.mod(y + vy*dt, Ly)
        if Inter:
            ax, ay, epot, virial = accelALL(x, y, Lx, Ly)
        vx += ax*dt/2
        vy += ay*dt/2
        guarda(it)

    return {'X': X, 'Y': Y, 'vx': vx, 'vy': vy, 't': dt*np.arange(Nt+1),
            'ekin': ekin, 'epot': Epot, 'virial': Vir, 'vcm': VCM}


def plot_trajetorias(res, particulas, Lx, Ly, titulo, ax=None):
    """Plots the x–y trajectories (particles numbered from 1)."""
    if ax is None:
        ax = plt.figure().gca()
    for p in particulas:
        ax.plot(res['X'][:, p-1], res['Y'][:, p-1], '.', ms=2, label='Parti: %d' % p)
    ax.set_xlim(0.0, Lx)
    ax.set_ylim(0.0, Ly)
    ax.set_aspect('equal')
    ax.set_xlabel('x(t)')
    ax.set_ylabel('y(t)')
    ax.set_title(titulo)
    ax.legend(loc='upper right', fontsize=8)
    return ax


def exercicio1(N=16, Lx=6.0, Ly=6.0, dt=0.01, Nt=200):
    """Free particles move in straight lines (wrapping around the box).
    With interactions the trajectories bend when two particles get close
    (repulsive collisions)."""
    x, y, vx, vy = ler_estado()
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, Inter, nome in zip(ax, (False, True), ('sem interações', 'com interações')):
        res = dinamica(x, y, vx, vy, Lx, Ly, dt, Nt, Inter)
        plot_trajetorias(res, (5, 6, 7, 10), Lx, Ly, 'Trajetórias: ' + nome, a)
    plt.tight_layout()
    plt.show()


def exercicio2(Lx=6.0, Ly=6.0, dt=0.01, passos=(1000, 2000)):
    """Time reversal. Velocity Verlet is exactly reversible in exact
    arithmetic, but round-off errors (~1e-16) are amplified exponentially by
    the chaotic dynamics. They are still small after 1000+1000 steps and
    become visible after 2000+2000 steps."""
    x0, y0, vx0, vy0 = ler_estado()
    fig, ax = plt.subplots(1, len(passos), figsize=(6*len(passos), 5))
    for a, Nt in zip(np.atleast_1d(ax), passos):
        ida = dinamica(x0, y0, vx0, vy0, Lx, Ly, dt, Nt)
        xf, yf = ida['X'][-1], ida['Y'][-1]
        if Nt == passos[0]:
            escrever_estado(os.path.join(PASTA, 'final.dat'), xf, yf, ida['vx'], ida['vy'])
        volta = dinamica(xf, yf, -ida['vx'], -ida['vy'], Lx, Ly, dt, Nt)
        # distance to init.dat (minimum image)
        dx = volta['X'][-1] - x0
        dy = volta['Y'][-1] - y0
        dx -= Lx*np.round(dx/Lx)
        dy -= Ly*np.round(dy/Ly)
        print("Nt = %d: distância máxima à configuração inicial após a inversão: %.3e"
              % (Nt, np.max(np.hypot(dx, dy))))
        plot_trajetorias(ida, (5, 6, 7), Lx, Ly, 'Nt = %d: ida (.) e volta (x)' % Nt, a)
        for p in (5, 6, 7):
            a.plot(volta['X'][::10, p-1], volta['Y'][::10, p-1], 'kx', ms=3)
    print("Configuração final (Nt = %d) escrita em final.dat" % passos[0])
    plt.tight_layout()
    plt.show()


def exercicio3a(Lx=6.0, Ly=6.0):
    """Relative energy error η(t) for (dt, Nt) = (0.01, 100) and (0.001, 1000).
    Without interactions E = E_kin is exactly constant. With interactions
    η oscillates and its amplitude scales as dt²."""
    x, y, vx, vy = ler_estado()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, Inter in zip(ax, (False, True)):
        for dt, Nt in ((0.01, 100), (0.001, 1000)):
            res = dinamica(x, y, vx, vy, Lx, Ly, dt, Nt, Inter)
            E = res['ekin'] + res['epot']
            eta = (E - E[0])/E[0]
            print("Interações=%s dt=%g: max|η| em [0,1] = %.3e" % (Inter, dt, np.max(np.abs(eta))))
            a.plot(res['t'], eta, label='dt = %g' % dt)
        a.set_title('η(t): ' + ('com' if Inter else 'sem') + ' interações')
        a.set_xlabel('t')
        a.set_ylabel('η(t)')
        a.legend()
    plt.tight_layout()
    plt.show()


def exercicio3b(Lx=6.0, Ly=6.0, dt=0.001, Nt=1000):
    """Conservation of the total momentum, and its correction."""
    x, y, vx, vy = ler_estado()
    plt.figure()
    for Inter in (False, True):
        res = dinamica(x, y, vx, vy, Lx, Ly, dt, Nt, Inter)
        mod = np.hypot(res['vcm'][:, 0], res['vcm'][:, 1])
        print("Interações=%s: |v_cm| inicial = %.6f, variação máxima = %.2e"
              % (Inter, mod[0], np.max(np.abs(mod - mod[0]))))
        plt.plot(res['t'], mod - mod[0], label=('com' if Inter else 'sem') + ' interações')
    vxc, vyc = corrige_vcm(vx, vy)
    print("init.dat tem v_cm = (%.4f, %.4f) ≠ 0, por isso precisa da correção;"
          " depois da correção v_cm = (%.1e, %.1e)" % (vcm(vx, vy) + vcm(vxc, vyc)))
    plt.title('|v_cm(t)| - |v_cm(0)|')
    plt.xlabel('t')
    plt.legend()
    plt.show()


def exercicio4(N=16, Lx=6.0, Ly=6.0, dt=0.01, Nt=10000):
    """Running time averages of the energies, temperature and pressure."""
    d = 2.0          # dimensão do sistema
    V = Lx*Ly        # "volume" (área) do sistema
    x, y, vx, vy = ler_estado()
    vx, vy = corrige_vcm(vx, vy)
    res = dinamica(x, y, vx, vy, Lx, Ly, dt, Nt)

    n = np.arange(1, Nt+2)
    Ekin_M = np.cumsum(res['ekin'])/n         # media no tempo usando todos os passos anteriores
    Epot_M = np.cumsum(res['epot'])/n
    Vir_M = np.cumsum(res['virial'])/n

    plt.figure()
    plt.plot(res['t'], Ekin_M/N, 'r-', label='Energia cinética')
    plt.plot(res['t'], Epot_M/N, 'b-', label='Energia potencial')
    plt.plot(res['t'], (Ekin_M + Epot_M)/N, 'k-', label='Energia total')
    plt.ylabel('<E>/N')
    plt.xlabel('t')
    plt.legend()
    plt.grid()

    T = (2.0/(d*N))*Ekin_M[-1]
    P = (2.0/(d*V))*Ekin_M[-1] + Vir_M[-1]/(d*V)
    rho = N/V
    print("Temperatura final: T = %.4f" % T)
    print("Pressão final:     P = %.4f" % P)
    print("Densidade:         ρ = N/V = %.4f" % rho)
    print("Gás ideal:         P = ρT = %.4f" % (rho*T))
    print("Efeito das interações (termo do virial): %+.4f" % (Vir_M[-1]/(d*V)))
    plt.show()


def grelha(N, Lx, Ly):
    """Positions on a regular √N × √N grid covering the Lx × Ly box."""
    n = int(round(np.sqrt(N)))
    assert n*n == N, "N tem de ser um quadrado perfeito"
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    return ((i.ravel() + 0.5)*Lx/n, (j.ravel() + 0.5)*Ly/n)


def vel_gerador(N, v0, rng):
    """Velocities of modulus v0 with a random direction in the plane."""
    a = rng.uniform(0, 2*np.pi, N)
    return v0*np.cos(a), v0*np.sin(a)


def exercicio5(v0=0.0, N=36, Lx=6.0, Ly=6.0, dt=0.01, Nt=10000, seed=0):
    """Grid start with |v| = v0: positions at it = 0, 100, Nt, and the
    trajectories of particles 4, 16 and 27."""
    rng = np.random.default_rng(seed)
    x, y = grelha(N, Lx, Ly)
    vx, vy = corrige_vcm(*vel_gerador(N, v0, rng))
    res = dinamica(x, y, vx, vy, Lx, Ly, dt, Nt)
    T = np.mean(res['ekin'])/N
    print("v0 = %g: temperatura média T = %.3f" % (v0, T))

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for it, cor in ((0, 'ro'), (100, 'ko'), (Nt, 'bo')):
        ax[0].plot(res['X'][it], res['Y'][it], cor, ms=4, label='it = %d' % it)
    ax[0].set_xlim(0, Lx)
    ax[0].set_ylim(0, Ly)
    ax[0].set_aspect('equal')
    ax[0].set_title('Posições (v0 = %g)' % v0)
    ax[0].legend(loc='upper right', fontsize=8)
    plot_trajetorias(res, (4, 16, 27), Lx, Ly, 'Trajetórias (v0 = %g)' % v0, ax[1])
    plt.tight_layout()
    plt.show()


def exercicio5a():
    exercicio5(v0=0.0)


def exercicio5b():
    exercicio5(v0=2.0)


# =============================================================================
# Part II — Diffusion Monte Carlo
# =============================================================================

def V(x):
    """Harmonic potential V(x) = x²/2."""
    return 0.5*np.square(x)


def phi(x):
    """Exact ground state φ0(x) = π^(-1/4) exp(-x²/2)."""
    return (1/np.pi**0.25)*np.exp(-np.square(x)/2)


def passo_dmc(x, E_R, dtau, N_P, alf, rng):
    """One DMC step: diffusion + branching + update of E_R."""
    x = x + rng.normal(0.0, 1.0, x.size)*np.sqrt(dtau)       # difusão
    q = np.exp(-(V(x) - E_R)*dtau)
    n = (q + rng.random(x.size)).astype(int)                 # número de cópias
    x = np.repeat(x, n)                                      # ramificação (nascimento/morte)
    E_R = E_R + alf*np.log(N_P/max(x.size, 1))
    return x, E_R


def funcao_onda(x, xmin=-5.0, xmax=5.0, Nbin=200):
    """Normalised wave function from the walker histogram (∫ψ² dx = 1)."""
    psi, bordas = np.histogram(x, bins=Nbin, range=(xmin, xmax))
    h = bordas[1] - bordas[0]
    centros = 0.5*(bordas[1:] + bordas[:-1])
    return centros, psi/np.sqrt(np.sum(psi**2)*h)


def dmc(x0, N_tau=2000, N_ter=10, dtau=0.01, E_R=0.0, N_P=3000, alf=0.3, seed=0):
    """Full DMC run: N_ter thermalisation steps (not counted in the averages)
    and then N_tau steps where E_R is accumulated."""
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    x, E_R = passo_dmc(x, E_R, dtau, N_P, alf, rng)
    x_primeiro = x.copy()                  # walkers after the very first time step
    for _ in range(N_ter - 1):
        x, E_R = passo_dmc(x, E_R, dtau, N_P, alf, rng)
    ER = np.empty(N_tau)
    for k in range(N_tau):
        x, E_R = passo_dmc(x, E_R, dtau, N_P, alf, rng)
        ER[k] = E_R
    return {'ER': ER, 'x': x, 'x_primeiro': x_primeiro, 'tau': dtau*np.arange(1, N_tau+1)}


def analisa_dmc(res, titulo):
    """Exercises 7–9 for one DMC run."""
    ER = res['ER']
    E_media = np.cumsum(ER)/np.arange(1, ER.size+1)       # <E(τ)> = S_E(τ)/k
    E_M = ER.mean()
    # Naive error: it ignores the correlation between consecutive steps, so it
    # underestimates the real error. The average also includes the first steps,
    # when E_R is still rising from 0, which pulls it slightly below 1/2.
    sig_E = np.sqrt((np.mean(ER**2) - E_M**2)/ER.size)
    print("%s: E0 = %.4f ± %.4f  (exato 0.5); passeantes no fim = %d"
          % (titulo, E_M, sig_E, res['x'].size))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(res['tau'], E_media, 'r-')
    ax[0].axhline(0.5, color='k', ls='--', label='E0 = 1/2')
    ax[0].set_ylabel('<E(τ)>')
    ax[0].set_xlabel('τ')
    ax[0].set_title(titulo)
    ax[0].legend()

    xs = np.linspace(-5, 5, 400)
    for x, nome in ((res['x_primeiro'], 'primeiro passo'), (res['x'], 'último passo')):
        c, psi = funcao_onda(x)
        ax[1].plot(c, psi, label='DMC: ' + nome)
    ax[1].plot(xs, phi(xs), 'k--', label='analítica')
    ax[1].set_ylabel('φ(x)')
    ax[1].set_xlabel('x')
    ax[1].legend(fontsize=8)
    plt.tight_layout()


def exercicio6(N_P=3000, seed=0):
    """Initial uniform distribution of the walkers in [-1, 1]."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, N_P)
    plt.figure()
    plt.hist(x, bins=100, range=(-1.2, 1.2), density=True)
    plt.title('Distribuição inicial aleatória')
    plt.ylabel('P(x)')
    plt.xlabel('x')
    plt.show()
    return x


def exercicio7_9(N_P=3000, seed=0):
    """Exercises 7, 8 and 9 starting from the uniform distribution."""
    rng = np.random.default_rng(seed)
    res = dmc(rng.uniform(-1, 1, N_P), seed=seed)
    analisa_dmc(res, 'Início uniforme em [-1, 1]')
    plt.show()


def exercicio10(N_P=3000, seed=0):
    """Repeats 7–9 with all walkers starting at x = 0 (a delta function).
    The result is the same: the ground state does not depend on the start."""
    res = dmc(np.zeros(N_P), seed=seed)
    analisa_dmc(res, 'Início em x = 0')
    plt.show()


EXERCICIOS = {
    '1': exercicio1,
    '2': exercicio2,
    '3a': exercicio3a,
    '3b': exercicio3b,
    '4': exercicio4,
    '5a': exercicio5a,
    '5b': exercicio5b,
    '6': exercicio6,
    '7': exercicio7_9,
    '10': exercicio10,
}


if __name__ == '__main__':
    args = sys.argv[1:]
    if args == ['dmc']:
        args = ['6', '7', '10']
    for chave in args or list(EXERCICIOS):
        if chave in ('8', '9'):      # 8 and 9 are produced together with 7
            chave = '7'
        print("\n" + "=" * 70 + "\nExercício " + chave + "\n" + "=" * 70)
        EXERCICIOS[chave]()
