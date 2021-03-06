import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import os

# argv 1: fewshot input, 2: normal input, 3: x_axis, 4: output file
# import all data files
folder = os.path.join(sys.argv[1], '')
data_files = [folder + "fewshot_results.txt", folder + "normal_results.txt"]
labels = ["Fewshot", "Normal"]
all_data = []
for file in data_files:
    with open(file) as f:
        lines = f.readlines()
        # change string in format "5.02e+02\n" to number
        for i in range(len(lines)):
            number_string = lines[i]
            number_string = number_string.replace('\n', '')
            lines[i] = float(number_string)
        all_data.append(lines)

# import x_axis
x_axis_file = folder + "result_weigths.txt"
x_axis = []
with open(x_axis_file) as f:
    lines = f.readlines()
    # change string in format "0.1\n" to number
    for i in range(len(lines)):
            number_string = lines[i]
            number_string = number_string.replace('\n', '')
            lines[i] = float(number_string)
    x_axis = lines

for i in range(len(all_data)):
    plt.plot(x_axis, all_data[i], label=labels[i])

plt.xlabel("Weight of pendulum")
plt.ylabel("Score")
plt.title("Fewshot vs normal learning")
plt.legend(loc="best")

path = Path(folder)
plt.savefig(folder + path.name + ".pdf")
