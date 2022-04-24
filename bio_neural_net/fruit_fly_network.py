#!/usr/bin/python3
import numpy as np
import os.path
import pandas as pd
from itertools import product, chain

from .network import Network

from .neuron import NeuronCluster, InputNeuronCluster

from .synapse import SynapseCluster, NMDASynapseCluster

PATH = os.path.dirname(os.path.realpath(__file__))

EB_INNERVATION = pd.read_csv(
        os.path.join(PATH, 'EB_innervation.csv'),
        index_col=0)
PB_INNERVATION = pd.read_csv(
        os.path.join(PATH, 'PB_innervation.csv'),
        index_col=0)

EIP_LABELS = [f'EIP{num}' for num in range(18)]
PEI_LABELS = [f'PEI{num}' for num in range(16)]
PEN_LABELS = [f'PEN{num}' for num in range(16)]

CONDUCTANCE_DICT = {
        ('EIP', 'PEI'): 5,
        ('EIP', 'PEN'): 6,
        ('EIP', 'REIP'): 1,
        ('PEI', 'EIP'): 4,
        ('PEN', 'EIP'): 6,
        ('REIP', 'EIP'): 5,
        ('REIP', 'REIP'): 1.6,
        ('RPEI', 'PEI'): 10,
        ('RPEN', 'PEN'): 10
}

DEFAULT_NEURON_PARAMS = {
    'size': 10,
    'Cm': 0.1,  # nF
    'VL': -70.0,  # mV
    'threshold': -50.0,  # mV
    'gL': 0.1e-9 / 15e-3 / 1e-9  # nS
}

REIP_PARAMS = {
    **DEFAULT_NEURON_PARAMS,
    'Cm': 0.01,  # nF
    'gL': 0.01e-9 / 15e-3 / 1e-9  # nS
}

GABAA_PARAMS = {
    'reversal_potential': -70.0,  # mV
    'time_constant': 0.005  # s
}

ACETYLCHOLINE_PARAMS = {
    'reversal_potential': 0.0,  # mV
    'time_constant': 0.020  # s
}

NMDA_PARAMS = {
    'reversal_potential': 0.0,  # mV
    'time_constant': 0.100  # s
}

INPUT_NEURONS = {
    **{name+'_input': {'size': 10, 'freq': 50}
       for name in EB_INNERVATION.columns},
    'rot_CW': {'size': 10, 'freq': 315},
    'rot_CCW': {'size': 10, 'freq': 315},
    'RPEN_input': {'size': 10, 'freq': 20},
    'RPEI_input': {'size': 10, 'freq': 20}
}

INPUT_SYNAPSE_CONDUCTANCE = {
    **{name+'_input': 2.1 for name in EB_INNERVATION.columns},
    'rot_CW': 0.3,
    'rot_CCW': 0.3,
    'RPEN_input': 10,
    'RPEI_input': 10
}

def get_fruit_fly_network(
            EIP_params=DEFAULT_NEURON_PARAMS,
            PEI_params=DEFAULT_NEURON_PARAMS,
            PEN_params=DEFAULT_NEURON_PARAMS,
            REIP_params=REIP_PARAMS,
            RPEI_params=DEFAULT_NEURON_PARAMS,
            RPEN_params=DEFAULT_NEURON_PARAMS,
            GABAA_params=GABAA_PARAMS,
            acetylcholine_params=ACETYLCHOLINE_PARAMS,
            NMDA_params=NMDA_PARAMS,
            conductance_dict=CONDUCTANCE_DICT,
            input_neurons=INPUT_NEURONS,
            input_synapse_conductance=INPUT_SYNAPSE_CONDUCTANCE
        ):
    net = Network()

    net.add_neurons(
        *(NeuronCluster(name, **EIP_params) for name in EIP_LABELS),
        *(NeuronCluster(name, **PEI_params) for name in PEI_LABELS),
        *(NeuronCluster(name, **PEN_params) for name in PEN_LABELS),
        NeuronCluster('RPEN', **RPEN_params),
        NeuronCluster('RPEI', **RPEI_params),
        NeuronCluster('REIP', **REIP_params),
        *(InputNeuronCluster(key, **params)
          for key, params in input_neurons.items())
    )

    # EIP, PEI, PEN connections
    for table in [EB_INNERVATION, PB_INNERVATION]:
        for src, trg in product(table.index, table.index):
            overlaps = sum(
                (table.loc[src] == 2) &
                (table.loc[trg] == 1))
            if overlaps == 0:  # no connections
                continue
            # if src[:3] == 'PEN' and trg in ['EIP0', 'EIP17']:
            #     overlaps = 3  # asterix in sup table 3
            factor = conductance_dict[(src[:3], trg[:3])]
            net.add_synapse(
                src,
                trg,
                NMDASynapseCluster(
                    max_conductance=overlaps * factor,
                    **NMDA_params))

    # EIP and REIP connections
    for name in EIP_LABELS:
        net.add_synapse(name,
                        'REIP',
                        NMDASynapseCluster(
                            max_conductance=conductance_dict[('EIP', 'REIP')],
                            **NMDA_params))
        net.add_synapse('REIP',
                        name,
                        NMDASynapseCluster(
                            max_conductance=conductance_dict[('REIP', 'EIP')],
                            **GABAA_params))

    net.add_synapse('REIP',
                    'REIP',
                    SynapseCluster(
                        max_conductance=conductance_dict[('REIP', 'REIP')],
                        **GABAA_params))

    # PEI and RPEI connections
    for name in PEI_LABELS:
        net.add_synapse('RPEI', name,
                        SynapseCluster(
                            max_conductance=conductance_dict[('RPEI', 'PEI')],
                            **GABAA_params))

    # PEN and RPEN connections
    for name in PEN_LABELS:
        net.add_synapse('RPEN', name,
                        SynapseCluster(
                            max_conductance=conductance_dict[('RPEN', 'PEN')],
                            **GABAA_params))

    # input connections
    for region in EB_INNERVATION.columns:
        for trg in EB_INNERVATION.loc[EB_INNERVATION[region] == 2].index:
            net.add_synapse(
                region+'_input',
                trg,
                SynapseCluster(
                    max_conductance=input_synapse_conductance[region+'_input'],
                    **acetylcholine_params))

    net.add_synapse(
        'RPEN_input',
        'RPEN',
        SynapseCluster(
            max_conductance=input_synapse_conductance['RPEN_input'],
            **acetylcholine_params))

    net.add_synapse(
        'RPEI_input',
        'RPEI',
        SynapseCluster(
            max_conductance=input_synapse_conductance['RPEI_input'],
            **acetylcholine_params))

    for trg in [f'PEN{num}' for num in range(8)]:
        net.add_synapse(
            'rot_CW',
            trg,
            SynapseCluster(
                max_conductance=input_synapse_conductance['rot_CW'],
                **acetylcholine_params))

    for trg in [f'PEN{num}' for num in range(8, 16)]:
        net.add_synapse(
            'rot_CCW',
            trg,
            SynapseCluster(
                max_conductance=input_synapse_conductance['rot_CCW'],
                **acetylcholine_params))

    return net
