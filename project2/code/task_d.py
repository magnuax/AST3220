import numpy as np
from tabulate import tabulate


H0 = 100*0.7
t0 = 13.7e12 # [yr]
T0 = 2.7260  # [K]

def t(T):
    return t0*(T0/T)**2


temps = np.array([1e8, 1e9, 1e10])
names = ["10^8", "10^9", "10^10"]
ages  = t(temps)*31556926           #[s]
headers = ["Temp. [K]", "Age [s]"]
table = tabulate(zip(names,ages), headers=headers, tablefmt="github")

print(table)
