#%% import header blocks
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, root
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import math

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200
})

#%% q1

# part a

print('q1 part a:')

ls = []

for k in range(100):
    for j in range(0, k + 1):
        ls.append(1.5 ** k * 2 ** j / math.factorial(k) / math.factorial(j))
        
print(f'P(team A wins) = {1 - np.exp(-3.5) * sum(ls)}') 

#part b

print('\nq1 part b:')

def p_1(t, k):
    ls = []
    
    for j in range(k + 1):
        A = (2 - 2 * t / 90) ** j
        B = (1.5 - 1.5 * t / 90) ** j
        C = math.factorial(j) ** 2
        
        ls.append(A * B / C)
        
    return np.exp(-3.5 +3.5 * t / 90) * sum(ls)

t = np.linspace(0, 90, 10000)
p = p_1(t, 50)

plt.figure(figsize=(7, 4))
plt.plot(t, p)
plt.xlabel('Time t (minutes)')
plt.ylabel('P(final tie | no goals up to t)')
plt.title('Probability the game ends in a tie')
plt.grid(alpha=0.3)
plt.tight_layout()
# plt.savefig('homework/hw7/q1pb.png')
plt.show()

# part c

print('\nq1 part c:')

def p_2(t, k):
    ls = []
    
    for j in range(k + 1):
        A = (2 - 2 * t / 90) ** j
        B = (1.5 - 1.5 * t / 90) ** (j + 1)
        C = math.factorial(j) * math.factorial(j + 1)
        ls.append(A * B / C)
        
    return np.exp(-3.5 + 3.5 * t / 90) * sum(ls)

t1 = np.linspace(0, 60-1e-100, 1000)
t2 = np.linspace(60, 90, 500)

p1 = p_1(t1, 50)
p2 = p_2(t2, 50)

plt.figure(figsize=(7, 4))
plt.plot(t1, p1, label=r't$\in$[0, 60)')
plt.plot(t2, p2, label=r't$\in$[60, 90]')

plt.annotate(
    'Team A scores a goal',
    xy=(60, 0.3),
    xytext=(5, 0.15),
    arrowprops={'arrowstyle': '->'}
)

plt.annotate(
    "Otis's life savings evaporating",
    xy=(78, 0.125),
    xytext=(6, 0.025),
    arrowprops={'arrowstyle': '->'}
)

plt.xlabel('Time t (minutes)')
plt.ylabel('P(final tie)')
plt.vlines(60, 0, 0.425, linestyles='--', colors='k', lw=0.8)
plt.title('Probability the game ends in a tie given team A scores at t = 60 min')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
# plt.savefig('homework/hw7/q1pc.png')
plt.show()

#%% q2

# part b

print('\nTesting Bessel function asymptotic approximation:')

from scipy.special import iv # Modified Bessel function of the first kind of real order

def p(t, lambd=1):
    return np.exp(-2 * lambd * t) * iv(0, 2 * lambd * t)

def p_asymptotic(t, lambd=1):
    return 1 / np.sqrt(4 * np.pi * lambd * t)

t = np.linspace(0.01, 4, 100000)
p_tie = p(t)
p_tie_asymptotic = p_asymptotic(t)

plt.figure(figsize=(10, 5))
plt.plot(t, p_tie, lw=1, label='modified bessel')
plt.plot(t, p_tie_asymptotic, lw=1, label='asymptotic approximation')

plt.xlabel('Time t')
plt.ylabel('P(final tie | no goals up to t)')
plt.title('Probability the game ends in a tie')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
# plt.savefig('homework/hw7/actual_vs_asymptotic.png')
plt.show()

# error of asymptotic approximation
rel_err = np.abs(p_tie - p_tie_asymptotic) / np.abs(p_tie)

plt.figure(figsize=(5, 3.5))
plt.plot(t, rel_err, lw=1)
plt.xlabel('Time t')
plt.ylabel('Relative error')
plt.title('Relative error of asymptotic approximation')
plt.grid(alpha=0.3)
plt.tight_layout()
# plt.savefig('homework/hw7/asymptotic_error.png')
plt.show()

# part c

print('\nq2 part c:')

T = 48

def inv_exp_CDF(y, lambd=3):
    return -np.log(1 - y) / lambd

T_A_ls = []
T_B_ls = []

# team A scoring times
tA = 0
while True:
    U_A = np.random.uniform(0, 1)
    tA += inv_exp_CDF(U_A)
    
    if tA > T:
        break
    
    T_A_ls.append(tA)

# team B scoring times
tB = 0
while True:
    U_B = np.random.uniform(0, 1)
    tB += inv_exp_CDF(U_B)
    
    if tB > T:
        break
    
    T_B_ls.append(tB)

number_of_baskets_scored_A = len(T_A_ls)
number_of_baskets_scored_B = len(T_B_ls)

plt.figure(figsize=(12, 5))

# red vertical bars for team A
plt.vlines(T_A_ls, 0.55, 1, color='red', label='Team A', linewidth=1.5)

# blue vertical bars for team B
plt.vlines(T_B_ls, 0, 0.45, color='blue', label='Team B', linewidth=1.5)

plt.xlim(0, T)
plt.ylim(-0.1, 1.1)
plt.xlabel('Time (minutes)')
plt.yticks([0.225, 0.775], ['Team B', 'Team A'])
plt.title('48-minute game')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
# plt.savefig('homework/hw7/q2pc.png')
plt.show()

print(f'Team A final score: {2 * number_of_baskets_scored_A}')
print(f'Team B final score: {2 * number_of_baskets_scored_B}')

# part d

print('\nq2 part d:')

lambd = 3

T_ls = []
T_A_ls = []
T_B_ls = []

t = 0
while True:
    U = np.random.uniform(0, 1)
    t += inv_exp_CDF(U, 2 * lambd)
    
    if t > T:
        break
    
    bernoli_RV = np.random.binomial(n=1, p=0.5)
    
    if bernoli_RV == 1:
        T_A_ls.append(t)
    else:
        T_B_ls.append(t)
    
number_of_baskets_scored_A = len(T_A_ls)
number_of_baskets_scored_B = len(T_B_ls)

plt.figure(figsize=(12, 5))

# red vertical bars for team A
plt.vlines(T_A_ls, 0.55, 1, color='red', label='Team A', linewidth=1.5)

# blue vertical bars for team B
plt.vlines(T_B_ls, 0, 0.45, color='blue', label='Team B', linewidth=1.5)

plt.xlim(0, T)
plt.ylim(-0.1, 1.1)
plt.xlabel('Time (minutes)')
plt.yticks([0.225, 0.775], ['Team B', 'Team A'])
plt.title('48-minute game')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
# plt.savefig('homework/hw7/q2pd.png')
plt.show()

print(f'Team A final score: {2 * number_of_baskets_scored_A}')
print(f'Team B final score: {2 * number_of_baskets_scored_B}')

# part e 

print('\nq2 part e:')

sim_runs = int(1e5)

lambd = 3
t = 48

# simulate total number of baskets in each game
N = np.random.poisson(2 * lambd * t, size=sim_runs)

# given total baskets N, assign each basket to team A with probability 1/2
N_A = np.random.binomial(N, 0.5)
N_B = N - N_A

# score difference
D = 2 * (N_A - N_B)

# simulated estimates
sim_mean = np.mean(D)
sim_var = np.var(D)
sim_p_tie = np.mean(D == 0)

# theoretical values
theory_mean = 0
theory_var = 8 * lambd * t
theory_p_tie = p(t, lambd)

# print results
print(f'\nSimulated mean = {sim_mean}')
print(f'Simulated variance = {sim_var}')
print(f'Simulated P(D(t)=0) = {sim_p_tie}')

print(f'\nTheoretical mean = {theory_mean}')
print(f'Theoretical variance = {theory_var}')
print(f'Theoretical P(D(t)=0) = {theory_p_tie}')

print(f'\nMean error = {abs(sim_mean - theory_mean)}')
print(f'Variance relative error = {abs(sim_var - theory_var) / theory_var}')
print(f'Tie probability error = {abs(sim_p_tie - theory_p_tie)}')

#%% q3

# part a

print('q3 part a:')
print('Numerical integration')

from scipy.integrate import quad

def integrand(t):
    return 0.5 * (t + t ** 3 / 900) * np.exp(-0.5 * (t + t ** 3 / 2700))

a, b = 0, np.inf

I = quad(integrand, a, b)

print(f'\nE[T_1]={I[0]}, error={I[1]}')

# part c

print('\nq3 part c:')

T = 120

def lambd(t):
    return 0.5 * (1 + (t / 30) ** 2)

lambda_max = lambd(T)   # monotonic function on [0,120], so max occurs at 120

# sample candidate arrival times from HPPP(lambda_max)
t = 0
T_ls = []

while True:
    t += np.random.exponential(1 / lambda_max)
    if t > T:
        break
    T_ls.append(t)

# keep arrival times T_i w.p. lambda(T_i)/lambda_max
T_i_ls = []
for ti in T_ls:
    p = lambd(ti) / lambda_max
    if np.random.random() <= p:
        T_i_ls.append(ti)

# histogram per day
bins = np.arange(0, T + 1, 1)
t_range = np.linspace(0, T, 1000)

plt.hist(T_i_ls, bins=bins, alpha=0.6, edgecolor='k', label='sampled reports/day')
plt.plot(t_range, lambd(t_range), lw=3, label='model rate $\\lambda(t)$')
plt.xlabel('Day')
plt.ylabel('Count / rate')
plt.title('Flu sympotoms reports per day')
plt.legend()
# plt.savefig('homework/hw7/q3pc.png')
plt.show()

print(f'simulated total reports = {len(T_i_ls)}')
print('expected total reports = 380')