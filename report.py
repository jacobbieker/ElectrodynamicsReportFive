import os, csv
import numpy as np
import matplotlib.pyplot as plt

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
for dataset in files[0][2]:
    file_count = 0
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


