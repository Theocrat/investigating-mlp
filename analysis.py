import sys
import numpy as np
import sklearn.datasets as ds
import matplotlib.pyplot as plt

# Settings
ACCEPTABLE_COMMAND_LINE_ARGUMENTS = {0, 1, 2, 3}

# Override default settings from command line arguments
try:
    feature_1 = int(sys.argv[1])
    feature_2 = int(sys.argv[2])
    
    ok_1 = feature_1 in ACCEPTABLE_COMMAND_LINE_ARGUMENTS
    ok_2 = feature_2 in ACCEPTABLE_COMMAND_LINE_ARGUMENTS
    args_ok = ok_1 and ok_2
    if not args_ok:
        raise(ValueError())   

except IndexError:
    print("There must be two arguments for X and Y axes", file=sys.stderr)
    exit(1)

except ValueError:
    print("Arguments must be integers in the range 0, 1, 2, 3", file=sys.stderr)
    exit(2)

# Loading the Iris dataset
iris = ds.load_iris()
data = iris["data"]
labels = iris["target"]

# Identifying the indices for each category
type_0 = labels == 0
type_1 = labels == 1
type_2 = labels == 2

# Slicing the data into each group
data_0 = data[type_0, :]
data_1 = data[type_1, :]
data_2 = data[type_2, :]

# Isolating features of interest
sepal_0 = data_0[:, [feature_1, feature_2]]
sepal_1 = data_1[:, [feature_1, feature_2]]
sepal_2 = data_2[:, [feature_1, feature_2]]

# Debugging
print("Feature Names:", iris["feature_names"])
print("Target Names:", iris["target_names"])
print(f"{sepal_0[:5, :] = }")

# Analytical plots
plt.xlabel(iris["feature_names"][feature_1])
plt.ylabel(iris["feature_names"][feature_2])
p0, *_ = plt.plot(sepal_0[:, 0], sepal_0[:, 1], "o", color="#4d68")
p1, *_ = plt.plot(sepal_1[:, 0], sepal_1[:, 1], "o", color="#d648")
p2, *_ = plt.plot(sepal_2[:, 0], sepal_2[:, 1], "o", color="#46d8")
plt.legend([p0, p1, p2], iris["target_names"])
plt.grid()
plt.tight_layout()
plt.show()