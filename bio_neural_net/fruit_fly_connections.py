'''
Tables from Su et al. Supplementary material.
https://www.nature.com/articles/s41467-017-00191-6
'''

import numpy as np
import pandas as pd

PB_labels = columns = [f'PB-R{i}' for i in range(8, -1, -1)] + [f'PB-L{i}' for i in range(9)]
EBC_labels = columns = [f'EB-R{i}C' for i in range(8, 0, -1)] + [f'EB-L{i}C' for i in range(1, 9)]
EBP_labels = columns = [f'EB-R{i}P' for i in range(8, 0, -1)] + [f'EB-L{i}P' for i in range(1, 9)]
EIP_labels = [f'EIP{i}' for i in range(18)]
PEI_labels = [f'PEI{i}' for i in range(16)]
PEN_labels = [f'PEN{i}' for i in range(16)]
tables = []

EIP_PB = pd.DataFrame(
            data = 2*np.eye(18),
            columns = PB_labels,
            index = EIP_labels)
tables.append(EIP_PB)

PEI_PB = pd.DataFrame(
            data = np.diag([1]*len(PB_labels))[1:-1],
            columns = PB_labels,
            index = PEI_labels)
tables.append(PEI_PB)

data = np.zeros((len(PEN_labels), len(PB_labels)))
data[:8, :8] = np.diag([1]*8)
data[8:, -8:] = np.diag([1]*8)
PEN_PB = pd.DataFrame(
            data = data,
            columns = PB_labels,
            index = PEN_labels)
tables.append(PEN_PB)

data = np.zeros((len(EIP_labels), len(EBC_labels)))
data[0, 0] = 1
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
tables.append(EIP_EBC)

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
tables.append(PEI_EBC)

EIP_EBP = pd.DataFrame(
            data = EIP_EBC.to_numpy(),
            columns = EBP_labels,
            index = EIP_labels)
tables.append(EIP_EBP)

PEN_EBP = pd.DataFrame(
            data = PEI_EBC.to_numpy(),
            columns = EBP_labels,
            index = PEN_labels)
tables.append(PEN_EBP)

data = np.zeros((len(PEI_labels), len(EIP_labels)))
for k in [-7, -8, 1, 9, 10]:
    data += np.diag(np.ones(len(EIP_labels) - abs(k)), k = k)[:len(PEI_labels)]
for row in range(8):
    data[row, row + 2] = 1
for row in range(8, 16):
    data[row, row] = 1
data[8, -2] = 1
data[7, 1] = 1

PEI_EIP = pd.DataFrame(
            data=data,
            columns=EIP_labels,
            index=PEI_labels)
tables.append(PEI_EIP)

data = np.zeros((len(PEN_labels), len(EIP_labels)))
for k in [-7, -8, 1, 9, 10]:
    data += np.diag(np.ones(len(EIP_labels) - abs(k)), k=k)[:len(PEI_labels)]
for row in range(8):
    data[row, row + 2] = 1
for row in range(8, 16):
    data[row, row] = 1
data[8, -2] = 1
data[7, 1] = 1
PEN_EIP = pd.DataFrame(
            data=data,
            columns=EIP_labels,
            index=PEN_labels)
tables.append(PEN_EIP)

# R_EIP to EIP

data = np.zeros((len(EIP_labels), len(PEI_labels)))
data[1:-1] = np.eye(16)
EIP_PEI = pd.DataFrame(
            data=data,
            columns=PEI_labels,
            index=EIP_labels)
tables.append(EIP_PEI)

# R_PEI to PEI

data = np.zeros((len(EIP_labels), len(PEN_labels)))
data[:8, :8] = np.eye(8)
data[-8:, -8:] = np.eye(8)
EIP_PEN = pd.DataFrame(
            data = data,
            columns = PEN_labels,
            index = EIP_labels)
tables.append(EIP_PEN)

# R_PEN to PEN

# tables on page 8+



del(data)


if __name__ == '__main__':
    for tab in tables:
        my_strs = tab.to_string().split('\n')
        print('-'*len(my_strs[0]))
        print(my_strs[0])
        for s in my_strs[1:]:
            print(s[:5], end='')
            print(s[5:].replace('0', ' ').replace('.', ' '))
        print('-'*len(my_strs[0]))
