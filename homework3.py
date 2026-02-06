#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200
})

#%%
#q1

x = np.linspace(0, 10, 100000)
a = np.sqrt(3) - 1

fx = ((x * (1 + x)) * (np.exp(-x))) / 3
gx = a ** 2 * x * np.exp(-a * x)
c = (3 * a ** 2 * np.exp(a) * (1 - a)) ** -1

plt.plot(x, fx, label = 'f(x)')
plt.plot(x, (c * gx), label = '$c(a^*)g_{a^*}(x)$')
plt.legend()
plt.show()

#%%
#q2

# part e
P = np.array([[9/10, 1/10,    0],
              [0,    7/8 ,  1/8],
              [2/5,   0,    3/5]])

def probability_matrix_multiplier(M, n): #perfomes matrix multiplication n many times
    M_cum = M
    for _ in range(1, n + 1):
        x = M_cum@M
        M_cum = x
    return M_cum

# print(probability_matrix_multiplier(P, 1000))

# part f

states = np.array(['G', 'S', 'D'], dtype = object)
P_df = pd.DataFrame(P, index=states, columns=states)

N = 100000
cancer_cell_state = 'G'          # start in G
sim = [cancer_cell_state]

for _ in range(N):
    probs = P_df.loc[cancer_cell_state].values   # row of probs (sums to 1)
    cancer_cell_state = np.random.choice(states, p=probs)
    sim.append(cancer_cell_state)

total_Gs = sim.count('G')

print(total_Gs / len(sim))
print(probability_matrix_multiplier(P, N))