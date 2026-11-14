# Física Computacional — Trabalhos 1 a 4 (Universidade de Coimbra)

Python projects from the Computational Physics course. The original code was
written in Python 2. It has been ported to Python 3, debugged, documented, and
each project is now gathered in a single module with one function per exercise.

| Folder | Module | Topics |
|---|---|---|
| `Trabalho 1 (FC)` | `trabalho1.py` | Finite square well (bisection, regula falsi, Newton-Raphson, secant), Newton interpolation, Gauss-Legendre quadrature, quadrupole moment of the hydrogen atom |
| `Trabalho 2 (FC)` | `trabalho2.py` | Resistor network (Gaussian elimination), RLC circuit (RK4), vibrations of a linear molecule, normal modes (Jacobi method) |
| `Trabalho 3 (FC)` | `trabalho3.py` | Vibrating string and 2D water waves (finite differences), 2D Ising model with Metropolis, compared with Onsager's exact solution |
| `Trabalho 4 (FC)` | `trabalho4.py` | Molecular dynamics of a 2D Lennard-Jones gas (velocity Verlet), Diffusion Monte Carlo for the harmonic oscillator |

The physics and the numerical method of each exercise are explained in the
docstring at the top of each module and in each function.
The assignment statements and the reports are the PDFs in each folder.

## Requirements

Python ≥ 3.8 with `numpy`, `scipy`, `matplotlib` and `pandas`:

```bash
pip install numpy scipy matplotlib pandas
```

Tested with Python 3.14, numpy 2.4, scipy 1.18, matplotlib 3.11.

## Usage

Run the module from inside its folder, optionally naming the exercises:

```bash
cd "Trabalho 1 (FC)"
python trabalho1.py          # every exercise
python trabalho1.py 4 9.3    # only exercises 4 and 9.3
```

Other options:

* `python trabalho2.py 7 --ci 3` runs exercise 7 with initial condition 3 only.
* `python trabalho3.py 8 --ns 2000` uses fewer Monte Carlo steps, so it runs faster.
* `python trabalho4.py dmc` runs only the Diffusion Monte Carlo part.
