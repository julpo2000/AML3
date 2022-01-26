import matplotlib.pyplot as plt
import numpy as np

filename = 'fitness.txt'
data = np.loadtxt(filename, delimiter=',', dtype=str)

elite_fit =[]
best_fit = []
peak_fit = []
index = []

for i in range(len(data)):
    cur_text = data[i].split()
    elite_fit.append(float(cur_text[5]))
    best_fit.append(float(cur_text[9]))
    peak_fit.append(float(cur_text[13]))
    index.append(i)



plt.plot(index, elite_fit, label="Elite")
plt.plot(index, best_fit, label="Best")
plt.plot(index, peak_fit, label="Peak")
plt.ylabel("Fitness")
plt.xlabel("Generation")
plt.legend()
plt.tight_layout()
plt.savefig("fitness.pdf")

