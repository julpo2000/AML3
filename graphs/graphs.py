import matplotlib.pyplot as plt
import numpy as np

filename = 'time.txt'
data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=str)

graph_data =[]

# prev = 0
# for i in range(len(data)):
#     dif = int(float(data[i])) - int(float(data[prev]))
#     graph_data.append(int(float(dif)))
#     prev = i

for i in range(len(data)):
    graph_data.append(int(float(data[i])))

plt.plot(graph_data)
plt.ylabel("Time in seconds")
plt.xlabel("Generation")
plt.tight_layout()
plt.savefig("time_total.pdf")

