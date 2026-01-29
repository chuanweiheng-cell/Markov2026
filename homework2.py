#%%
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200
})

#%%
#q1
def f(x, x0=10, gamma=4):
    C = (gamma - 1) * x0 ** (gamma - 1)
    return C * x ** -gamma
    
def f_inv(x, x0=10, gamma=4):
    return (x0 ** (1 - gamma) - x0 ** (1 - gamma) * x) ** (1 / (1 - gamma))

Ns = [100, 1000, 10000, 100000, 1000000]

all_samples = []
for i in range(len(Ns)):
    samples = []
    for _ in range(Ns[i]):
        U = np.random.uniform(0, 1)
        X = f_inv(U)        
        samples.append(X)
    all_samples.append(samples)

for i in range(len(Ns)):
    gamma = 4
    x0 = 10
    x_range = np.linspace(x0, 60, 100000)
    pdf = f(x_range)
    plt.hist(all_samples[i], bins = 60, range = (0, 60), density=True)
    plt.plot(x_range, pdf, label = 'pdf')
    plt.legend()
    plt.title(f'N={Ns[i]}')
    plt.show()
    
#%%
#q2
from scipy.optimize import brentq
import time

#part b

def F(x):
    return 1 - (x + 1) * np.exp(-x)

ls_b = []

start_time = time.perf_counter()

for _ in range(10 ** 6):
    U = np.random.uniform(0, 1)
    sample = brentq(lambda x: F(x) - U, 0, 10000)
    ls_b.append(sample)
    
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Run time for q2 part b: {elapsed_time:.4f} seconds")

# part c

def f(x):
    return x * np.exp(-x)

def g(x):
    return np.exp(-x / 2) / 2

c = 4 / np.e

def G_inv(x):
    return -2 * np.log(1 - x)

ls_c = []
accepted = 0

start_time = time.perf_counter()

while accepted < 10 ** 6:
    U1 = np.random.uniform(0, 1)
    U2 = np.random.uniform(0, 1)
    X = G_inv(U1)
    
    if U2 <= f(X) / c / g(X):
        ls_c.append(X)    
        accepted += 1
        
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Run time for q2 part c: {elapsed_time:.4f} seconds")

# part d

ls_d = []

start_time = time.perf_counter()

for _ in range(10 ** 6):
    U1 = np.random.exponential(1)
    U2 = np.random.exponential(1)
    gamma = U1 + U2
    
    ls_d.append(gamma)

end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Run time for q2 part d: {elapsed_time:.4f} seconds")

# part e

x_range = np.linspace(0, 15, 1000000)
gamma_dist = f(x_range)

plt.hist(ls_b, bins = 50, density=True)
plt.plot(x_range, gamma_dist, label='Gamma Distribution')
plt.legend()
plt.title('Method from part b')
plt.show()

plt.hist(ls_c, bins = 50, density=True)
plt.plot(x_range, gamma_dist, label='Gamma Distribution')
plt.legend()
plt.title('Method from part c')
plt.show()

plt.hist(ls_d, bins = 50, density=True)
plt.plot(x_range, gamma_dist, label='Gamma Distribution')
plt.legend()
plt.title('Method from part d')
plt.show()