#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from fruit_fly import (
    Network,
    NeuronCluster,
    InputNeuron,
    DEFAULT_NEURON_PARAMS,
    REIP_PARAMS,
    SynapseCluster,
    NMDA,
    NMDA_PARAMS,
    GABAA_PARAMS,
)
from fruit_fly.connections import (
    EIP_labels,
    PEI_labels,
    PEN_labels,
    PEI_EIP,
    PEN_EIP,
    EIP_PEI,
    EIP_PEN
)

from itertools import product

# times in ms
TIME_START = 0.0
TIME_FINAL = 1000 * 10.0  # adjusted to match step size
STEP_SIZE = 1e-3

steps = int((TIME_FINAL - TIME_START)/STEP_SIZE)
ts = (np.arange(steps)-TIME_START) * STEP_SIZE

net = Network(TIME_START, STEP_SIZE)

net.add(*(
    NeuronCluster(name, 10, **DEFAULT_NEURON_PARAMS)
    for name in (EIP_labels + PEI_labels + PEN_labels +
                 ['REIP', 'RPEN', 'RPEI'])
))

#######################
# fix conductances?
#######################

for table in [
        PEI_EIP,
        PEN_EIP,
        EIP_PEI,
        EIP_PEN]:
    for col, row in product(table.columns, table.index):
        if table[col][row]:
            NMDA(net[row],
                 net[col],
                 max_conductance=1,  # fix me!!!
                 **NMDA_PARAMS)

# EIP and REIP connections
for name in EIP_labels:
    NMDA(net[name], net['REIP'], max_conductance=1, **NMDA_PARAMS)
    SynapseCluster(net['REIP'], net[name], max_conductance=5, **GABAA_PARAMS)

SynapseCluster(net['REIP'], net['REIP'], max_conductance=1.6, **GABAA_PARAMS)

# PEI and RPEI connections
for name in PEI_labels:
    SynapseCluster(net['RPEI'], net[name], max_conductance=10, **GABAA_PARAMS)


# PEN and RPEN connections
for name in PEN_labels:
    SynapseCluster(net['RPEN'], net[name], max_conductance=10, **GABAA_PARAMS)

# add input neurons

#######################
# main
#######################
if __name__ == '__main__':
    # net.print_full()
    net.reset()
    for step in tqdm(range(steps)):
        net.update()
