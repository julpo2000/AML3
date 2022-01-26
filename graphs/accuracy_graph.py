import matplotlib.pyplot as plt
import numpy as np

accuracy ={
    4096 : 0.5,
    2048 : 0.3
}

lists = sorted(accuracy.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.ylabel("Accuracy")
plt.xlabel("Generation")
plt.tight_layout()
plt.savefig("accuracy_per_gen.pdf")