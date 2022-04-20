#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from bio_neural_net import (
    Network,
    NeuronCluster,
    InputNeuronCluster,
    DEFAULT_NEURON_PARAMS,
    REIP_PARAMS,
    SynapseCluster,
    NMDASynapseCluster,
    NMDA_PARAMS,
    GABAA_PARAMS,
)
from connections import (
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
TIME_FINAL = 10.0  # adjusted to match step size
STEP_SIZE = 1e-7

steps = int((TIME_FINAL - TIME_START)/STEP_SIZE)
ts = (np.arange(steps)-TIME_START) * STEP_SIZE

net = Network(TIME_START, STEP_SIZE, steps)

net.add_neurons(*(
    NeuronCluster(name, 10, **DEFAULT_NEURON_PARAMS)
    for name in (EIP_labels + PEI_labels + PEN_labels +
                 ['REIP', 'RPEN', 'RPEI'])
))

#######################
# Add input neurons
#######################

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
            net.add_synapse(
                 row,
                 col,
                 NMDASynapseCluster(
                     max_conductance=1,  # fix me!!!
                     **NMDA_PARAMS))

# EIP and REIP connections
for name in EIP_labels:
    net.add_synapse(name,
                    'REIP',
                    NMDASynapseCluster(
                        max_conductance=1,
                        **NMDA_PARAMS))
    net.add_synapse('REIP',
                    name,
                    SynapseCluster(
                        max_conductance=5,
                        **GABAA_PARAMS))

net.add_synapse('REIP', 'REIP',
                SynapseCluster(max_conductance=1.6, **GABAA_PARAMS))

# PEI and RPEI connections
for name in PEI_labels:
    net.add_synapse('RPEI', name,
                    SynapseCluster(max_conductance=10, **GABAA_PARAMS))


# PEN and RPEN connections
for name in PEN_labels:
    net.add_synapse('RPEN', name,
                    SynapseCluster(max_conductance=10, **GABAA_PARAMS))

# add input neurons

#######################
# main
#######################
if __name__ == '__main__':
    # net.print_full()
    net.reset()
    for step in tqdm(range(steps)):
        net.update()
