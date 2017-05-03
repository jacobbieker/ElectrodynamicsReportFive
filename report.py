import os, csv, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import numpy.polynomial.polynomial as poly

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

error_ind = 0.0001
error_res = 1
error_freq = 0.1

total_error_x = np.sqrt(error_res**2 + error_res**2 + error_ind**2 + error_ind**2)
total_error_y = np.sqrt(error_freq ** 2 + error_freq**2)

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
    return 1/(L*(w_nought**2))

def get_gamma(R):
    return R/2

def equation(X, R):
    # A is 150/?
    # A2Γ2/(4(ω−ω0)2+Γ2/4)
    w, L, w_nought = X
    w = np.asarray(w)
    top = ((150/R)**2)*((R/L)**2)/4
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
    print("FWHM:" + str(FWHM(vm_vlc, w_nought)))

    vm_vlc = [x for (y, x) in sorted(zip(w_nought, vm_vlc))]
    w_nought = sorted(w_nought)
    plt.scatter(w_nought,vm_vlc)

    w_nought = w_nought
    vm_vlc = vm_vlc

    index = w_nought.index(max(w_nought))
    p0 = [500]
    print(curve_fit(equation, (w_nought, num_inductors*0.0022, w_nought[index]), vm_vlc))
    popt, pcov = curve_fit(equation, (w_nought, num_inductors*0.0022, w_nought[index]), vm_vlc)
    print(w_nought)
    print("C: " + str(get_c(num_inductors*0.0022, w_nought[index])))
    print("R: " + str(popt))
    min_val = min(w_nought)
    fit_line = np.linspace(min_val, max(w_nought), num=100)
    y_fit = []
    for element in fit_line:
        y_fit.append(equation((element, num_inductors*0.0022, w_nought[index]), popt))

    plt.plot(fit_line, y_fit)
    return w_nought[index], popt, get_c(num_inductors*0.0022, w_nought[index])

Rc = []
w0 = []
co = []
Rlc = []
Rl = []
one, rlc, c = create_graph(data[0][0], data[0][1], data[0][2], 1)
two, rc, d = create_graph(data_no[0][0], data_no[0][1], data_no[0][2], 1)
plt.title("One Inductor")
plt.ylabel("(Vm/Vlc)^2")
plt.xlabel("w0")
print("Rc 1 Inductor: " + str(rlc - rc))
Rc.append((rlc - rc)[0])
Rlc.append(rlc[0])
Rl.append(rc[0])
w0.append(one)
co.append(c)
plt.show()

one, rlc, c = create_graph(data[1][0], data[1][1], data[1][2], 2)
two, rc, d = create_graph(data_no[1][0], data_no[1][1], data_no[1][2], 2)
plt.title("Two Inductors")
plt.ylabel("(Vm/Vlc)^2")
plt.xlabel("w0")
print("Rc: " + str(rlc - rc))
Rc.append((rlc - rc)[0])
w0.append(one)
Rlc.append(rlc[0])
Rl.append(rc[0])
co.append(c)
plt.show()

one, rlc, c = create_graph(data[2][0], data[2][1], data[2][2], 3)
two, rc, d = create_graph(data_no[2][0], data_no[2][1], data_no[2][2], 3)
plt.title("Three Inductors")
plt.ylabel("(Vm/Vlc)^2")
plt.xlabel("w0")
print("Rc: " + str(rlc - rc))
Rc.append((rlc - rc)[0])
w0.append(one)
Rlc.append(rlc[0])
Rl.append(rc[0])
co.append(c)
plt.show()

one, rlc, c = create_graph(data[3][0], data[3][1], data[3][2], 4)
two, rc, d = create_graph(data_no[3][0], data_no[3][1], data_no[3][2], 4)
plt.title("Four Inductors")
plt.ylabel("(Vm/Vlc)^2")
plt.xlabel("w0")
print("Rc: " + str(rlc - rc))
Rc.append((rlc - rc)[0])
w0.append(one)
Rlc.append(rlc[0])
Rl.append(rc[0])
co.append(c)
plt.show()

one, rlc, c = create_graph(data[4][0], data[4][1], data[4][2], 5)
two, rc, d = create_graph(data_no[4][0], data_no[4][1], data_no[4][2], 5)
plt.title("Five Inductors")
plt.ylabel("(Vm/Vlc)^2")
plt.xlabel("w0")
print("Rc: " + str(rlc - rc))
Rc.append((rlc - rc)[0])
w0.append(one)
Rlc.append(rlc[0])
Rl.append(rc[0])
co.append(c)
plt.show()

print(w0)
print(Rc)
print(c)

#plt.scatter(w0, Rc)
plt.errorbar(w0, Rc, yerr=total_error_y, xerr=total_error_x)
plt.xlabel("w0")
plt.ylabel("Rc")
plt.show()

total_error_x = np.sqrt(error_ind**2 + error_freq**2 + error_freq**2 )
total_error_y = error_freq**2
#plt.scatter(w0, co)
plt.errorbar(w0, co, yerr=total_error_y, xerr=total_error_x)
plt.xlabel("w0")
plt.ylabel("C")
plt.show()

print("Rlc: ")
print(Rlc)
print("Rl: ")
print(Rc)