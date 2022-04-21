#!/usr/bin/python3
import numpy as np

from .network import Network

from .neuron import (
    NeuronCluster,
    InputNeuronCluster,
    DEFAULT_NEURON_PARAMS,
    REIP_PARAMS,
)

from .synapse import (
    SynapseCluster,
    NMDASynapseCluster,
    NMDA_PARAMS,
    GABAA_PARAMS,
    ACETYLCHOLINE_PARAMS
)

from .fruit_fly_connections import (
    EIP_labels,
    PEI_labels,
    PEN_labels,
    EIP_EBC,
    PEI_EBC,
    EIP_EBP,
    PEN_EBP,
    EIP_PB,
    PEI_PB,
    PEN_PB,
    PEI_EIP,
    PEN_EIP,
    EIP_PEI,
    EIP_PEN
)

from itertools import product

def get_fruit_fly_network():
    net = Network()

    EB_inputs = [f'EB-R{num}_input' for num in range(8, 0, -1)] + \
                [f'EB-L{num}_input' for num in range(1, 9)]

    net.add_neurons(
        *(NeuronCluster(name, 10, **DEFAULT_NEURON_PARAMS)
          for name in (EIP_labels + PEI_labels + PEN_labels +
                       ['REIP', 'RPEN', 'RPEI'])),
        *(InputNeuronCluster(name, 10, 50)
          for name in EB_inputs),
        InputNeuronCluster('rot_CW', 10, 3150),
        InputNeuronCluster('rot_CCW', 10, 3150),
        InputNeuronCluster('RPEN_input', 10, 200),
        InputNeuronCluster('RPEI_input', 10, 200)
    )

    # synapses
    for table, src, trg, conductance_factor in [
            (EIP_PEI, EIP_PB, PEI_PB, 5),
            (EIP_PEN, EIP_PB, PEN_PB, 6),
            (PEI_EIP, PEI_EBC, EIP_EBC, 4),
            (PEN_EIP, PEN_EBP, EIP_EBP, 6)]:

        for col, row in product(table.columns, table.index):
            if table[col][row]:
                overlaps = sum(
                    (src.loc[row] == 2) &
                    (trg.loc[col] == 1))
                if overlaps == 0:
                    print(f'No overlaps between {row}, {col}')
                if table is PEN_EIP and row in ['EIP0', 'EIP17']:
                    overlaps = 3  # asterix in sup table 3
                net.add_synapse(
                     row,
                     col,
                     NMDASynapseCluster(
                         max_conductance=conductance_factor*overlaps,
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

    # input connections
    for src in EB_inputs:
        src_lookup = src[:-6] + 'C'
        for trg in EIP_EBC.loc[EIP_EBC[src_lookup] == 1].index:
            net.add_synapse(src, trg,
                            SynapseCluster(max_conductance=2.1,
                                           **ACETYLCHOLINE_PARAMS))

    net.add_synapse('RPEN_input', 'RPEN',
                    SynapseCluster(max_conductance=10,
                                   **ACETYLCHOLINE_PARAMS))

    net.add_synapse('RPEI_input', 'RPEI',
                    SynapseCluster(max_conductance=10,
                                   **ACETYLCHOLINE_PARAMS))

    for trg in [f'PEN{num}' for num in range(8)]:
        net.add_synapse('rot_CW', trg,
                        SynapseCluster(max_conductance=0.3,
                                       **ACETYLCHOLINE_PARAMS))

    for trg in [f'PEN{num}' for num in range(8, 16)]:
        net.add_synapse('rot_CCW', trg,
                        SynapseCluster(max_conductance=0.3,
                                       **ACETYLCHOLINE_PARAMS))

    return net
