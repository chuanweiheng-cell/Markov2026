#%% import header blocks
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, root
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200
})

#decomposition dictionary
DECOMPOSITIONS = {
    'eig': lambda A: np.linalg.eig(A),                 # (eigenvalues, eigenvectors)
    'eigh': lambda A: np.linalg.eigh(A),               # (eigenvalues, eigenvectors) for symmetric/Hermitian
    'svd': lambda A: np.linalg.svd(A, full_matrices=False),  # (U, S, Vt)
    'qr': lambda A: np.linalg.qr(A),                   # (Q, R)
    'cholesky': lambda A: np.linalg.cholesky(A),       # L such that A = L @ L.T
}

def decompose(A, method):
    A = np.asarray(A, dtype=float)
    method = method.strip().lower()

    if method not in DECOMPOSITIONS:
        raise ValueError(f'Unknown method: {method}. Choose from {list(DECOMPOSITIONS.keys())}.')

    return DECOMPOSITIONS[method](A)

SCALAR_METHODS = {
    'bisect': 'bisect',
    'brentq': 'brentq',
    'brenth': 'brenth',
    'ridder': 'ridder',
    'newton': 'newton',
    'secant': 'secant'
}

#%% q1

#part b

print('part b:')

P = np.array([[0, 1, 0, 0, 0],
              [1/3, 0, 2/3, 0, 0],
              [0, 1/2, 0, 1/2, 0],
              [0, 0, 2/3, 0, 1/3],
              [0, 0, 0, 1, 0]])

evals, evecs = decompose(P.T, 'eig')

print(f'\neugenvalues:\n{evals}')
print(f'eigenvectors (columns):\n{evecs}')
idx = np.argmin(np.abs(evals - 1)) # returns index of eigenvale 1
v = np.real(evecs[:, idx])
pi = v / v.sum()
print(f'\npi={pi}')

#part d

print('\npart d:')


q0 = np.array([0, 0, 1, 0, 0])

q50 = q0.T @ np.linalg.matrix_power(P, 50)

print(f'\nq50={q50}')

xvals = []
yvals = []
pivals = []

for i in range(0, q50.size):
    xvals.append(i)
    yvals.append(q50[i])
    pivals.append(pi[i])
    
plt.plot(xvals, yvals, marker='o', label=r'$\vec{q}_{50}(i)$')
plt.plot(xvals, pivals, marker='o', label=r'$\vec{\pi}(i)$')
plt.legend()
plt.xlabel('i')
plt.ylabel('Probability')
plt.title(r'Stationary distribution $\vec{\pi}$ vs $\vec{q}_{50}$')
# plt.savefig('homework/hw6/q1pd.png', dpi=500)
plt.show()

#%% q3

def nroot_scalar(f, method, bracket=None, x0=None, x1=None, fprime=None, tol=1e-10, max_iter=1000):
    
    method = method.strip().lower()

    if method not in SCALAR_METHODS:
        raise ValueError(f'Unknown scalar method: {method}. Choose from {list(SCALAR_METHODS.keys())}.')

    kwargs = {
        'method': SCALAR_METHODS[method],
        'xtol': tol,
        'maxiter': max_iter
    }

    if bracket is not None:
        kwargs['bracket'] = bracket
    if x0 is not None:
        kwargs['x0'] = x0
    if x1 is not None:
        kwargs['x1'] = x1
    if fprime is not None:
        kwargs['fprime'] = fprime

    sol = root_scalar(f, **kwargs)
    return sol.root, sol.iterations, sol.converged

def find_all_roots_scalar(f, method, guesses, fprime=None, tol=1e-10, max_iter=1000, decimals=10):
    
    method = method.strip().lower()

    if method not in SCALAR_METHODS:
        raise ValueError(f'Unknown scalar method: {method}. Choose from {list(SCALAR_METHODS.keys())}.')

    roots = []

    for guess in guesses:
        try:
            if method in ['bisect', 'brentq', 'brenth', 'ridder']:
                root_val, _, converged = nroot_scalar(
                    f,
                    method,
                    bracket=guess,
                    tol=tol,
                    max_iter=max_iter
                )

            elif method == 'newton':
                if fprime is None:
                    raise ValueError('Newton method requires fprime.')
                root_val, _, converged = nroot_scalar(
                    f,
                    method,
                    x0=guess,
                    fprime=fprime,
                    tol=tol,
                    max_iter=max_iter
                )

            elif method == 'secant':
                root_val, _, converged = nroot_scalar(
                    f,
                    method,
                    x0=guess[0],
                    x1=guess[1],
                    tol=tol,
                    max_iter=max_iter
                )

            else:
                continue

            if converged:
                roots.append(root_val)

        except Exception:
            pass

    roots = np.unique(np.round(roots, decimals))
    return roots

# part a

print('part a:\n')

lmbd = 1/2
f = lambda x: np.exp((x - 1) * lmbd) - x
df = lambda x: lmbd * np.exp((x - 1) * lmbd) - 1

guesses = np.linspace(-2, 2, 50)
roots_newton = find_all_roots_scalar(f, 'newton', guesses, fprime=df)
print('roots found with newton =', roots_newton)

# part b

print('\npart b:\n')

lmbd = 1
f = lambda x: np.exp((x - 1) * lmbd) - x
df = lambda x: lmbd * np.exp((x - 1) * lmbd) - 1

guesses = np.linspace(-2, 2, 50)
roots_newton = find_all_roots_scalar(f, 'newton', guesses, fprime=df)
print('roots found with newton =', roots_newton)
print(f'rho = min solution = {min(roots_newton)}')

# part c

print('\npart c:\n')

lmbd = 2
f = lambda x: np.exp((x - 1) * lmbd) - x
df = lambda x: lmbd * np.exp((x - 1) * lmbd) - 1

guesses = np.linspace(-2, 2, 50)
roots_newton = find_all_roots_scalar(f, 'newton', guesses, fprime=df)
print('roots found with newton =', roots_newton)
print(f'rho = min solution = {min(roots_newton)}')

#%% q4

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

a = 0.49

r1 = (1 + np.sqrt(1 - 4 * (1 - a) * a)) / 2 / (1 - a)
r2 = (1 - np.sqrt(1 - 4 * (1 - a) * a)) / 2 / (1 - a)

rho = min(r1, r2) # P(extinction)

simulated_avalanches = int(1e4)
runs = 200
ext = 0

plot_every = 20 # store every 20th trajectory
trajectories = []

for trial in range(simulated_avalanches):
    Xn = 1   # initialize, X0 = 1
    Xn_ls = [Xn]
    count = 0

    while (count < runs) and (Xn > 0):
        Zn = np.random.binomial(Xn, 1 - a)   # number of successful reproductions
        Xn = 2 * Zn   # each reproducer makes 2 children
        Xn_ls.append(Xn)
        count += 1

    if Xn == 0:
        ext += 1

    # store every plot_every-th trajectory for animation
    if trial % plot_every == 0:
        trajectories.append(Xn_ls)

frac = ext / simulated_avalanches
print(f'Simulated probability of extinction = {frac}')
print(f'Theoretical probability of extinction = {rho}')


# plotting/animation 

fig, ax = plt.subplots()

max_n = max(len(path) for path in trajectories) - 1
max_x = max(max(path) for path in trajectories)

ax.set_xlim(0, max_n)
ax.set_ylim(0, max_x + 1)
ax.set_title(f'Avalanche simulation, P(offspring = 0) = {a}')
ax.set_xlabel('n')
ax.set_ylabel('$X_n$')

def update(frame):
    path = trajectories[frame]
    n_range = np.arange(len(path))

    ax.plot(n_range, path, '-o', lw=0.5, ms=1.5)
    return ax.lines


animation = FuncAnimation(fig, 
                    update, 
                    frames=len(trajectories), 
                    interval=50, 
                    repeat=False)

plt.show()
# animation.save('homework/hw6/q4pb_animation.mp4')
# plt.savefig('homework/hw6/q4pb.png')          
# HTML(animation.to_jshtml())

#%% q5

from scipy.special import binom

theoratical_prob =(-1) ** 5 * binom(1/2, 2)

a = 0.5

simulated_avalanches = int(1e4)
runs = 2
size_3_avalanche = 0

plot_every = 20 # store every 20th trajectory
trajectories = []

for trial in range(simulated_avalanches):
    Xn = 1   # initialize, X0 = 1
    Xn_ls = [Xn]
    count = 0

    while (count < runs) and (Xn > 0):
        Zn = np.random.binomial(Xn, 1 - a)   # number of successful reproductions
        Xn = 2 * Zn   # each reproducer makes 2 children
        Xn_ls.append(Xn)
        count += 1

    Sn = sum(Xn_ls)

    if Sn == 3:
        size_3_avalanche += 1

    # store every plot_every-th trajectory for animation
    if trial % plot_every == 0:
        trajectories.append(Xn_ls)

frac = size_3_avalanche / simulated_avalanches
print(f'Simulated probability of a size 3 avalanche = {frac}')
print(f'Theoratical probability of a size 3 avalanche = {theoratical_prob}')

# plotting/animation 

fig, ax = plt.subplots()

max_n = max(len(path) for path in trajectories) - 1
max_x = max(max(path) for path in trajectories)

ax.set_xlim(0, max_n)
ax.set_ylim(0, max_x + 1)
ax.set_title(f'Avalanche simulation')
ax.set_xlabel('n')
ax.set_ylabel('$X_n$')

def update(frame):
    path = trajectories[frame]
    n_range = np.arange(len(path))

    ax.plot(n_range, path, '-o', lw=0.5, ms=1.5)
    return ax.lines


animation = FuncAnimation(fig, 
                    update, 
                    frames=len(trajectories), 
                    interval=50, 
                    repeat=False)

plt.show()
# animation.save('homework/hw6/q4pb_animation.mp4')
# plt.savefig('homework/hw6/q4pb.png')          
# HTML(animation.to_jshtml())