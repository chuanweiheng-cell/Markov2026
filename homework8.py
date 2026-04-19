#%% import header blocks
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200
})

#%% q1

# part a

print('q1 part a')

Q = np.array([
    [-1, 1, 0, 0],
    [0, -1, 1, 0],
    [0, 0, -1, 1],
    [1, 0, 0, -1]
])

A = Q.T

vals, vecs = np.linalg.eig(A)

print('\nQ^T =')
print(A)
print('\neigenvalues =')
print(np.round(vals, decimals=5))
print('\neigenvectors of Q^T (columns) =')
print(np.round(vecs, decimals=6))

# stationary distribution: eigenvector corresponding to eigenvalue 0
pi = vecs[:, 3].real
pi = pi / np.sum(pi)

print(f'\nStationary distribution pi = {np.round(pi, 6)}')

# part d

print('\nq1 part d')

y0 = np.array([1/3, 2/3, 0, 0], dtype=complex)

# coefficients in eigenvector basis
C = np.linalg.solve(vecs, y0)
print(f'\nc = {np.round(C, 6)}')

# coefficients appearing in y1(t)
first_entries = vecs[0, :]
final_coeffs = C * first_entries

print(f'\nCoefficients in y1(t) = {np.round(final_coeffs, 6)}')

# part e

print('\nq1 part e')

states = np.array(['1', '2', '3', '4'])
state_to_index = {state: i for i, state in enumerate(states)}

y0_probs = np.array([1/3, 2/3, 0, 0])

Ns = [100, 1000, 10_000, 100_000]
t_final = 5
t_grid = np.linspace(0, t_final, 1000)

# theoretical solution for y1(t) from part d
f_theory = np.zeros_like(t_grid, dtype=complex)

for k in range(len(vals)):
    f_theory += final_coeffs[k] * np.exp(vals[k] * t_grid)

plt.plot(t_grid, f_theory, color='k', lw=3, label='Theoratical')

for N in Ns:
    state_1_counts = np.zeros_like(t_grid, dtype=float)

    for i in range(N):
        current_state = np.random.choice(states, p=y0_probs)

        time_elapsed = 0
        jump_times = [time_elapsed]
        state_history = [current_state]

        while True:
            idx = state_to_index[current_state]
            rate = -Q[idx, idx]
            scale = 1 / rate
            tau = np.random.exponential(scale=rate) # transition opccures at time tau

            # stop if next jump would occur after t_final
            if time_elapsed + tau > t_final:
                break

            time_elapsed += tau

            probs = Q[idx].copy()
            probs[idx] = 0
            probs = probs / rate

            next_state = np.random.choice(states, p=probs)
            current_state = next_state

            jump_times.append(time_elapsed)
            state_history.append(current_state)

        jump_times = np.array(jump_times)
        state_history = np.array(state_history)

        # for each t in t_grid, find which state the chain is in
        interval_idx = np.searchsorted(jump_times, t_grid, side='right') - 1

        state_1_counts += (state_history[interval_idx] == '1')

    f_mc = state_1_counts / N

    plt.plot(t_grid, f_mc, lw=1.2, label=f'N = {N}')

plt.xlim(0, 5)
plt.ylim(0, 0.5)
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Fraction of chains in state 1')
plt.grid()
plt.legend()
# plt.savefig('homework/hw8/q1pe')
plt.show()

#%% q2

# part c

print('q2 part c')

Q = np.array([
    [-1, 1, 0],
    [1, -2, 1],
    [0, 1, -1]
])

A = Q.T

vals, vecs = np.linalg.eigh(A)

print('\nQ^T =')
print(np.round(A, decimals=5))
print('\neigenvalues =')
print(np.round(vals, decimals=5))
print('\neigenvectors (columns) =')
print(np.round(vecs, decimals=5))

y0 = np.array([1, 0, 0])

C = np.linalg.solve(vecs, y0)
print(f'\nc = {np.round(C, 6)}')

# coefficients appearing in y1(t)
second_entries = vecs[1, :]
final_coeffs = C * second_entries

print(f'\nCoefficients in y_F(t) = {np.round(final_coeffs, 6)}')

#%% q3

 # part b
 
print('\nq3 part b')
 
lambd = 1/2
nu = 1
mu = 1/12

N = 10000
ar = np.zeros(N)

for n in range(N):
    pi_k = 1
    for k in range(1, n + 1):
        term = lambd / (nu + k * mu)
        pi_k *= term
    ar[n] = pi_k

pi0 = 1 / sum(ar)

print(f'\npi0 = {pi0}')
pi = pi0 * ar
print(f'\nStationary distribution pi = {pi}')

# part d

print('\nq3 part d')

print(f'\nE[sales per hour] = {60 * (1 - pi0)}')

#%% q4

 # part a
 
print('\nq4 part a')

print('\nAnalyze the one matrix:')

N = 10

one = np.ones((N, N))

vals, vecs = np.linalg.eig(one)

print('\neigenvalues of one =')
print(np.round(vals, decimals=5))

NI = N * np.eye(N)

vals, vecs = np.linalg.eig(NI)

print('\neigenvalues of N*I =')
print(np.round(vals, decimals=5))

Q = one - NI

vals, vecs = np.linalg.eig(Q)

print('\neigenvalues of Q =')
print(np.round(vals, decimals=5))

# part b

print('\nq4 part b')

Q = np.array([
    [-2, 1, 0, 1],
    [1, -2, 1, 0],
    [0, 1, -2, 1],
    [1, 0, 1, -2]
])

vals, vecs = np.linalg.eig(Q)

print('\neigenvalues of Q =')
print(np.round(vals, decimals=5))