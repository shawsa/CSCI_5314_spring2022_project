import numpy as np
import pandas as pd

PB_labels = columns = [f'PB-R{i}' for i in range(8, -1, -1)] + [f'PB-L{i}' for i in range(9)]
EBC_labels = columns = [f'EB-R{i}C' for i in range(8, 0, -1)] + [f'EB-L{i}C' for i in range(1, 9)]
EBP_labels = columns = [f'EB-R{i}P' for i in range(8, 0, -1)] + [f'EB-L{i}P' for i in range(1, 9)]
EIP_labels = [f'EIP{i}' for i in range(18)]
PEI_labels = [f'PEI{i}' for i in range(16)]
PEN_labels = [f'PEN{i}' for i in range(16)]

EIP_PB = pd.DataFrame(
            data = 2*np.eye(18),
            columns = PB_labels, 
            index = EIP_labels)

PEI_PB = pd.DataFrame(
            data = np.diag([1]*len(PB_labels))[1:-1],
            columns = PB_labels,
            index = PEI_labels)

data = np.zeros((len(PEN_labels), len(PB_labels)))
data[:8,:8] = np.diag([1]*8)
data[8:, -8:] = np.diag([1]*8)
PEN_PB = pd.DataFrame(
            data = data,
            columns = PB_labels,
            index = PEN_labels)

data = np.zeros((len(EIP_labels), len(EBC_labels)))
data[0,0] = 1
data[-1, -1] = 1
for row in range(1, 9):
    data[row, (2*row - 2) % len(EBC_labels)] = 1
    data[row, (2*row - 1) % len(EBC_labels)] = 1
    data[row, (2*row - 0) % len(EBC_labels)] = 1

for row in range(9, 17):
    data[row, (2*row - 3) % len(EBC_labels)] = 1
    data[row, (2*row - 2) % len(EBC_labels)] = 1
    data[row, (2*row - 1) % len(EBC_labels)] = 1

EIP_EBC = pd.DataFrame(
            data = data,
            columns = EBC_labels,
            index = EIP_labels)

data = np.zeros((len(PEI_labels), len(EBC_labels)))
for row in range(8):
    data[row, (2*row+1)%len(EBC_labels)] = 2
    data[row, (2*row+2)%len(EBC_labels)] = 2
data[8] = data[7]
data[9:] = data[:7]

PEI_EBC = pd.DataFrame(
            data = data,
            columns = EBC_labels,
            index = PEI_labels)

PEN_EBC = pd.DataFrame(
            data = np.zeros((len(PEN_labels), len(EBC_labels))),
            columns = EBC_labels,
            index = PEN_labels)

EIP_EBP = pd.DataFrame(
            data = EIP_EBC.to_numpy(),
            columns = EBP_labels,
            index = EIP_labels)

PEI_EBP = pd.DataFrame(
            data = np.zeros((len(PEI_labels), len(EBP_labels))),
            columns = EBP_labels,
            index = PEI_labels)

PEN_EBP = pd.DataFrame(
            data = PEI_EBC.to_numpy(),
            columns = EBP_labels,
            index = PEN_labels)








del(data)
