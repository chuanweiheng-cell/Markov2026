#%% import header blocks
import numpy as np
import matplotlib.pyplot as plt
import math

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200
})

#%% q1

print('q1 part d')

alpha = beta = 1
L = 20

# State space is {0, 1, ..., L}, so there are L+1 states
Q = np.zeros((L + 1, L + 1))

# state 0
Q[0, 1] = alpha
Q[0, 0] = -alpha

# interior states 1, 2, ..., L-1
for row in range(1, L):
    Q[row, row - 1] = beta
    Q[row, row] = -(alpha + beta)
    Q[row, row + 1] = alpha

# state L is absorbing, so row L stays all zeros

states = np.array([str(i) for i in range(L + 1)], dtype=object)

# build dictionary automatically from states
state_to_index = {state: i for i, state in enumerate(states)}

# initial distribution, starts at the nucleas
initial_distribution = np.zeros(L + 1)
initial_distribution[0] = 1

N = 10000

# store full path information for each simulation
all_jump_times = []
all_state_histories = []

# store occupation fractions for each simulation
occupation_fractions = np.zeros((N, len(states)))

hitting_times = []

for i in range(N):
    current_state = np.random.choice(states, p=initial_distribution)

    time_elapsed = 0
    jump_times = [0]
    state_history = [current_state]

    time_in_state = np.zeros(len(states))

    while True:
        idx = state_to_index[current_state]

        # End simulation when we hit the delivery site
        if current_state == '20':
            break

        rate = -Q[idx, idx]
        scale = 1 / rate
        tau = np.random.exponential(scale=scale)

        # spend tau units of time in current state
        time_in_state[idx] += tau
        time_elapsed += tau

        # jump probabilities to other states
        probs = Q[idx, :].copy()
        probs[idx] = 0
        probs = probs / rate

        next_state = np.random.choice(states, p=probs)

        current_state = next_state
        
    hitting_times.append(time_elapsed)
    
print(f'\nAverage simulated transport time: {sum(hitting_times) / N:.02f}s')
print(f'Theoratical transport time: {L * (L + 1) / 2 / alpha}s')
print(f'Variance from simulation: {np.var(hitting_times)}')

#%% q3

print('\nq3')

lambd = 1/15
nu = 1/10

def SD(N):
    pi_0 = 1 / sum((lambd / nu) ** n / math.factorial(n) for n in range(N + 1))
    pi_N = (lambd / nu) ** N / math.factorial(N) * pi_0
    return pi_N

for N in range(1, 6):
    print(f'Try N={N}: pi_{N}={SD(N)}')
    
print('\nThus, number of parking spots needed is N=4')

#%% q4

print('\nq4')

def deterministic(m, beta=1):
    return np.log(m) / beta

def stochastic(m, beta=1):
    return np.array([sum(1 / k for k in range(1, n)) for n in m]) / beta

m = np.arange(1, 100)

y1 = stochastic(m)
y2 = deterministic(m)

plt.plot(m, y1, label='stochastic')
plt.plot(m, y2, label='deterministic')
plt.xlabel('m')
plt.ylabel('value')
plt.legend()
plt.grid()
# plt.savefig('homework/hw9/q4.png')
plt.show()