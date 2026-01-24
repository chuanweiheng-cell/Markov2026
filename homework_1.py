#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import random

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200
})

#%%

#q3

def f(u): 
    return u ** 4 / (1 + u ** 6)

ls = [1 + 0.1 * i for i in range(41)]

N = [np.floor(10 ** x) for x in ls]

fractions = []

for i in N:
    above = 0
    bellow = 0
    for j in range(int(i)):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        
        if f(x) >= y:
            bellow += 1
        else:
            above += 1
        
    frac = bellow / (bellow + above)
    fractions.append(frac)

horizontal_axis = np.log(N)

u = np.linspace(0, 1, 100000)
y = f(u)

soln = trapezoid(y, u)
print(soln)

plt.plot(horizontal_axis, fractions, '-o', lw = 1, label = 'Monte Carlo E(N)')
plt.axhline(soln, color = 'g', lw = 0.5, label = f'Numerical Intergration value = {soln:.3g}')
plt.legend()
plt.xlabel('log(N)')
plt.ylabel('Fraction of f(u) above sample point')
plt.title('Monte Carlo simukation vs Numerical Intergration')
plt.show()

#%%
#q4

N = 10 ** 5
Ts = []

for _ in range(N):
    T_M = random.uniform(0, 1)
    T_A = random.uniform(0, 1)
    T_D = random.uniform(0, 1)
    Ts.append(max(T_M, T_A, T_D))

plt.hist(Ts, bins = 50, density = True, edgecolor = 'k', label= 'Simulated')

t = np.linspace(0, 1, N)
plt.plot(t, 3 * t ** 2, linewidth = 2, label='Theory $f_T(t)=3t^2$')

plt.xlabel("T (hours after 6PM)")
plt.ylabel("pdf")
plt.legend()
plt.show()