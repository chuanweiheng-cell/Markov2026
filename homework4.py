#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import comb

#%%

#%% q1

import numpy as np

# part c
q = 0.4
p = 0.35
s = 0.25

N = 100_000

i_vals = []

for _ in range(N):
    i = 10
    U = s + 1  

    while (U > s) and (i > 0):
        U = np.random.uniform(0, 1)

        # retire
        if U < s:
            break

        # win (+1)
        elif U < s + p:
            i += 1

        # lose (-1)
        else:
            i -= 1

    # game ended: either retired (break) or ruined (i==0)
    i_vals.append(i)

mc_est = np.mean(i_vals)

# theoretical from part (b):
r2 = (1 - np.sqrt(1 - 4 * p * q)) / (2 * p)
theory = 10 + (p - q)/s * (1 - r2 ** 10)

print('Simulated estimate:', mc_est)
print('Theoretical value:', theory)
print('Percent error:', abs((mc_est - theory) / theory))

#%% q2

# part a

p = 1 / 10          # failure prob
q = 1 - p           # survival prob

P = np.zeros((6, 6), dtype=float)

P[0, 5] = 1

for i in range(1, 6):
    for j in range(0, i + 1):
        P[i, j] = comb(i, j) * (q ** j) * (p ** (i - j))

print(P)

#part b

#%% q2

# part a

p = 1 / 10          # failure prob
q = 1 - p           # survival prob

P = np.zeros((6, 6), dtype=float)

P[0, 5] = 1

for i in range(1, 6):
    for j in range(0, i + 1):
        P[i, j] = comb(i, j) * (q ** j) * (p ** (i - j))

print(P)

#part b

A = np.zeros((5, 5), dtype=float) 
b = np.ones(5, dtype=float)        
for i in range(1, 6):    
    row = i - 1

    A[row, i-1] = 1

    for j in range(1, 6): 
        A[row, j - 1] -= P[i, j]

E = np.linalg.solve(A, b) 

print('e1..e5 =', E)
print('Expected time starting from 5:', E[4])

# part c

A = P.T - np.eye(6)
b = np.zeros(6)

# Replace one row with the normalization equation sum(pi)=1
A[-1, :] = 1
b[-1] = 1

pi = np.linalg.solve(A, b)

print('stationary $pi$ =', pi)
print('sum(pi) =', pi.sum())
print('Long-run P(X=1) =', pi[1])