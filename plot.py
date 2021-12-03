import matplotlib.pyplot as plt
import numpy as np

# import all data files
data_files = ['results/32 gens/fewshot_results.txt', 'results/32 gens/normal_results.txt']
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
x_axis_file = 'results/32 gens/seq.txt'
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
plt.legend(loc="lower right")
plt.savefig("plots/Fewshot_normal_pendulum.pdf")
