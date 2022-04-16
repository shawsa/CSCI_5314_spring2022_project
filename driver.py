import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fruit_fly import (
        Network,
        NeuronCluster,
        NeuronType,
        SynapseType,
        SynapseCluster,
        InputSynapseCluster
        )

INPUT_CURRENT = -20

TIME_START = 0.0
TIME_FINAL = 50.0
STEP_SIZE = 1e-3

CUTOFF_TIME = 20


neuron_params = {
    'Cm': 0.2,
    'gL': 0.01,
    'VL': -70.0,
    'threshold': -50.0
}

synapse_params = {
    'time_constant': 3.0,
    'max_conductance': 0.005,
    'reversal_potential': 0.0
}

net = Network()

net.add(NeuronCluster('n1', 2, **neuron_params))
net.add(NeuronCluster('n2', 3, **neuron_params))
net.add(NeuronCluster('n3', 3, **neuron_params))

syn1 = InputSynapseCluster(net['n1'], INPUT_CURRENT)

syn2 = SynapseCluster(net['n1'], net['n2'], **synapse_params)
syn3 = SynapseCluster(net['n2'], net['n3'], **synapse_params)
syn4 = SynapseCluster(net['n3'], net['n1'], **synapse_params)

if __name__ == '__main__':
    print(net)
    steps = int((TIME_FINAL - TIME_START)/STEP_SIZE)
    ts = (np.arange(steps)-TIME_START) * STEP_SIZE
    n1_list = []
    n2_list = []
    n3_list = []
    s1_list = []
    s2_list = []
    s3_list = []
    s4_list = []
    for t in tqdm(ts):
        if t > CUTOFF_TIME:
            syn1.output_current = 0
        n1_list.append(np.average(net.neurons['n1'].V))
        n2_list.append(np.average(net.neurons['n2'].V))
        n3_list.append(np.average(net.neurons['n3'].V))
        s1_list.append(np.average(syn1.conductance(None)))
        s2_list.append(np.average(syn2.conductance(net['n2'].V)))
        s3_list.append(np.average(syn3.conductance(net['n3'].V)))
        s4_list.append(np.average(syn4.conductance(net['n1'].V)))
        net.update(STEP_SIZE)

    fig, axes = plt.subplots(3, 1, figsize=(15, 7))
    axes[0].plot(ts, n1_list, label='n1')
    axes[0].plot(ts, n2_list, label='n2')
    axes[0].plot(ts, n3_list, label='n3')
    axes[0].set_ylabel('mV')
    axes[1].plot(ts, s1_list, 'k:', label=r'$\mapsto n1$')
    axes[1].plot(ts, s2_list, label=r'$n1 \to n2$')
    axes[1].plot(ts, s3_list, label=r'$n2 \to n3$')
    axes[1].plot(ts, s4_list, label=r'$n3 \to n1$')
    axes[1].set_ylabel('mA')
    axes[1].set_xlabel('t (ms)')
    axes[1].tick_params(labelbottom=True)
    axes[0].legend()
    axes[1].legend()

    graph = nx.DiGraph()
    graph.add_nodes_from(net.nodes())
    graph.add_edges_from(net.edges())
    graph_attr = {
            'node_color': 'w',
            'edgecolors': 'k',  # node border color
            'with_labels': True
    }
    nx.draw(graph, ax=axes[2], **graph_attr)
    plt.show()
