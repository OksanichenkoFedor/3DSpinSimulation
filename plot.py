import matplotlib.pyplot as plt
import numpy as np

def plot(filename):
    C = np.loadtxt(filename)
    plt.figure(figsize=(10,7))
    plt.plot(C)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.show()
plot("29928_25000.txt")
