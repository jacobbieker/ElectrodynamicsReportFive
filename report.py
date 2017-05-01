import os, csv, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

data = []
data1 = []
data2 = []
data3 = []
data4 = []
data5 = []

data_no = []
data1_no = []
data2_no = []
data3_no = []
data4_no = []
data5_no = []

inductor = 2.2 #milliHenry
resistor = 150 #Ohm resistor

files = list(os.walk("data"))
file_count = 0
for dataset in files[0][2]:
    with open(os.path.join("data", dataset), "r") as csv_file:
        data.append([[],[],[]])
        data_no.append([[],[],[]])
        frequency = []
        vm = []
        vlc = []
        count = 0
        spamreader = csv.reader(csv_file, delimiter=',')
        for line in spamreader:
            count += 1
            if count < 13 and line != []:
                data[file_count][0].append([line[0]])
                data[file_count][1].append([line[1]])
                data[file_count][2].append([line[2]])
            elif line != []:
                data_no[file_count][0].append([line[0]])
                data_no[file_count][1].append([line[1]])
                data_no[file_count][2].append([line[2]])
    file_count += 1

def get_c(L, w_nought):
    return 1/(L*w_nought**2)

def get_gamma(R):
    return R/2

def equation(R, Rm, L, w_nought, w):
    # A is 150/?
    # A2Γ2/(4(ω−ω0)2+Γ2/4)
    top = ((Rm/R)**2)*((R/L)**2)/4
    bottom = (w - w_nought)**2 + ((R/L)**2)/4
    return top/bottom

#  Create graphs
def create_graph(frequency, vm, vlc, num_inductors):
    w_nought = []
    for element in frequency:
        element = float(element[0])
        w_nought.append(element/(2*np.pi))

    vm_vlc = []
    for index, element in enumerate(vm):
        element = float(element[0])
        bottom = float(vlc[index][0])
        vm_vlc.append((element/bottom)**2)

    def FWHM(X, Y):
        d = []
        for element in Y:
            d.append(element - (max(Y) / 2))
        d = np.asarray(d)
        indexes = np.where(d > 0)[0]
        return abs(X[indexes[-1]] - X[indexes[0]])

    print(w_nought)
    print("FWHM:" + str(FWHM(vm_vlc, w_nought)))
    print(vm_vlc)

    plt.scatter(w_nought,vm_vlc)

create_graph(data[0][0], data[0][1], data[0][2], 1)
create_graph(data_no[0][0], data_no[0][1], data_no[0][2], 1)
plt.title("One Inductor")
plt.ylabel("(Vm/Vlc)^2")
plt.xlabel("w0")
plt.show()

create_graph(data[1][0], data[1][1], data[1][2], 2)
create_graph(data_no[1][0], data_no[1][1], data_no[1][2], 2)
plt.title("Two Inductors")
plt.ylabel("(Vm/Vlc)^2")
plt.xlabel("w0")
plt.show()

create_graph(data[2][0], data[2][1], data[2][2], 3)
create_graph(data_no[2][0], data_no[2][1], data_no[2][2], 3)
plt.title("Three Inductors")
plt.ylabel("(Vm/Vlc)^2")
plt.xlabel("w0")
plt.show()

create_graph(data[3][0], data[3][1], data[3][2], 4)
create_graph(data_no[3][0], data_no[3][1], data_no[3][2], 4)
plt.title("Four Inductors")
plt.ylabel("(Vm/Vlc)^2")
plt.xlabel("w0")
plt.show()

create_graph(data[4][0], data[4][1], data[4][2], 5)
create_graph(data_no[4][0], data_no[4][1], data_no[4][2], 5)
plt.title("Five Inductors")
plt.ylabel("(Vm/Vlc)^2")
plt.xlabel("w0")
plt.show()



