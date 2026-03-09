#%% import header block
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize_scalar

import time

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200
})
#%% q2

# part b

print('part b:')

a = 0.04
b = 0.16
K = 0.1

def pn(n, K=K, a=a):
    return K * np.exp(a * n)

def qn(n, K=K, b=b):
    return K * np.exp(b * (n - 1))

ls_p = []
ls_q = []

for i in range(1, 5):
    ls_p.append(pn(i))
    
for j in range(2, 6):
    ls_q.append(qn(j))
    
p1, p2, p3, p4 = ls_p
q2, q3, q4, q5 = ls_q

P = np.array([[1 - p1, p1, 0, 0, 0],
              [q2, 1 - p2 - q2, p2, 0, 0],
              [0, q3, 1 - p3 - q3, p3, 0],
              [0, 0, q4, 1 - p4 - q4, p4],
              [0, 0, 0, q5, 1 - q5]])

# check if rows sum to 1
print('\ncheck if rows sum to 1:', P.sum(axis=1))

# eigen-decomposition of P^T
evals, evecs = np.linalg.eig(P.T)

print('\nEigenvalues of P^T:')
print(evals)

idx = np.argmin(np.abs(evals - 1))

v = evecs[:, idx]
v = np.real(v)

if v.sum() < 0:
    v = -v

# normalize
pi = v / v.sum()

print('\nStationary distribution pi:')
print(pi)
print('Sum(pi) =', pi.sum())
print('max|piP - pi| =', np.max(np.abs(pi @ P - pi)))
print('Long-run P(X=1) =', pi[0])

# part c

print('\n\npart c:')

N = int(1e6)

states = np.array(['1', '2', '3', '4', '5'], dtype=object)
P_df = pd.DataFrame(P, index=states, columns=states)

current_state = np.random.choice(states)
sim = []

for _ in range(N):
    probs = P_df.loc[current_state].values
    next_state = np.random.choice(states, p=probs)
    sim.append(next_state)
    current_state = next_state

# fraction of time in each state
counts = pd.Series(sim).value_counts().reindex(states, fill_value=0)
fractions = counts / len(sim)

print('\nCounts:')
print(counts)
print('\nFractions:')
print(fractions)

plt.figure()
plt.bar(states, fractions.values)
plt.xlabel("State")
plt.ylabel("Fraction of time spent")
plt.title(f"Empirical state fractions over {len(sim):,} steps")
plt.show()

print('\nStationary pi from part (b):', pi)
print('Empirical fractions:', fractions.values)
print('Abs error:', np.abs(fractions.values - pi))

# part d

print('\n\npart d')

# theoretical pi from part a
c = a - b  

weights = np.array([
    1,
    np.exp(c),
    np.exp(3 *c),
    np.exp(6 * c),
    np.exp(10 * c),
], dtype=float)

pi_a = weights / weights.sum()

print('\npi from (a) (detailed balance):', pi_a)
print('pi from (b) (eigenvector)     :', pi)
print('empirical fractions (c)       :', fractions.values)

# plot all 3 together

x = np.arange(1, 6)

plt.figure()
plt.bar(x- 0.25, pi_a, width=0.25, label='Theory (a)')
plt.bar(x, pi, width=0.25, label='Theory (b)')
plt.bar(x + 0.25, fractions.values.astype(float), width=0.25, label='Empirical (c)')

plt.xticks(x, ['1','2','3','4','5'])
plt.xlabel('State')
plt.ylabel('Probability / fraction of time')
plt.title('Empirical vs theoretical stationary distributions')
plt.legend()
plt.show()

print('\nmax|emp - a|:', np.max(np.abs(fractions.values.astype(float) - pi_a)))
print('max|emp - b|:', np.max(np.abs(fractions.values.astype(float) - pi)))
print('max|a - b|  :', np.max(np.abs(pi_a - pi)))

#%% q3

# part c

print('part c:')

a = 0.99 

A = np.array([[1-a, a, 0],
              [a , 0, 1-a],
              [0, 1-a, a]]) # Probability transition matrix

q0 = np.array([1, 0, 0]) # initial state vector

DECOMPOSITIONS = {
    'eig': lambda A: np.linalg.eig(A), 
    'eigh': lambda A: np.linalg.eigh(A),
    'svd': lambda A: np.linalg.svd(A, full_matrices=False),
    'qr': lambda A: np.linalg.qr(A),
    'cholesky': lambda A: np.linalg.cholesky(A),
} # dictionary of methods

def decompose(A, method): # matrix decomposition solver I made last time
    A = np.asarray(A, dtype=float)
    method = method.strip().lower()
    
    if method not in DECOMPOSITIONS:
        raise ValueError(f'Unknown method: {method}. Choose from {list(DECOMPOSITIONS.keys())}.')

    return DECOMPOSITIONS[method](A)

e_vals, e_vecs = decompose(A, 'eigh') # eigen values and vectors
print('\neigenvalues:\n', e_vals)
print('\neigenvectors (columns):\n', e_vecs)

def matrix_solver(A, b):
    return np.linalg.solve(A, b)

coefs = matrix_solver(e_vecs, q0) # coeficients of q0 expressed as as span{v1, v2, v3}

print(f'\nCoeficients of decompositions: \n{coefs}')

q0_recon = e_vecs @ coefs # test to see if we can recostruct q0 via span{v1, v2, v3}

np.set_printoptions(suppress=True, precision=6) # stops scientific notation for small numbers 

print('\nq0 original:', q0)
print('\nq0 reconstructed:', q0_recon)

# part d

print('\n\npart d:')

def lam(a):
    return 3 * a ** 2 - 3 * a + 1

# minimize over 0 < a < 1
res = minimize_scalar(lam, bounds=(0, 1), method='bounded')

a_star = res.x
lam_min = res.fun

print('\na* =', a_star)
print('lambda_min =', lam_min)

# part e

start_time = time.perf_counter()

print('\n\npart e:')

def matrix_power_recursive(M, n):  # recursive matrix multiplier
    if n < 0:
        raise ValueError('n must be a nonnegative integer')
    if n == 0:
        return np.eye(M.shape[0], dtype=M.dtype)
    if n == 1:
        return M

    # repeated squaring
    if n % 2 == 0:
        half_power = matrix_power_recursive(M, n // 2)
        return half_power @ half_power
    else:
        return M @ matrix_power_recursive(M, n - 1)

def qn_1(A, q0, n, i=0):  # generates theoretical q_n(i) value
    qn = matrix_power_recursive(A, n) @ q0
    return qn[i]

n_max = 500
n_range = np.arange(n_max + 1)

# theoretical q_n(1)
q_range = [qn_1(A, q0, n) for n in n_range]

states = np.array(['1', '2', '3'])
state_to_index = {'1': 0, '2': 1, '3': 2}

N_range = [100, 1000, 10_000]

for N in N_range:
    counts_1 = np.zeros(n_max + 1, dtype=int)

    for _ in range(N):
        current_state = '1'
        counts_1[0] += 1

        for n in range(1, n_max + 1):
            probs = A[state_to_index[current_state], :]
            current_state = np.random.choice(states, p=probs)
            if current_state == '1':
                counts_1[n] += 1

    f_n = counts_1 / N

    plt.figure(figsize=(8, 4))
    plt.plot(n_range, f_n, label=f'sim: N={N}', color='r')
    plt.plot(n_range, q_range, color='c', alpha=0.5, lw=2.5, label='theory: $q_{n}(1)$')

    plt.xlabel('n')
    plt.ylabel('prob / fraction in state 1')
    plt.title(f'Part (e): empirical vs theory (N={N})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Run time: {elapsed_time:.4f} seconds")