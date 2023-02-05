import numpy

data = [np.genfromtxt("./archive/Sample_"+i+"1.csv", delimiter=",") for i in range(0,5)]

print(data)