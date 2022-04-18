import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fruit_fly import (
    Network,
    NeuronCluster,
    SynapseCluster,
    InputSynapseCluster
)

INPUT_CURRENT = -20
INPUT_SD = .05

TIME_START = 0.0
TIME_FINAL = 50.0
STEP_SIZE = 1e-2

CUTOFF_TIME = 30


neuron_params = {
    'Cm': 0.2,
    'gL': 0.001,
    'VL': -70.0,
    'threshold': -50.0
}

synapse_params = {
    'time_constant': 3.0,
    'max_conductance': 0.001,
    'reversal_potential': 0.0
}

net = Network()

net.add(NeuronCluster('n1', 10, **neuron_params))
net.add(NeuronCluster('n2', 10, **neuron_params))
net.add(NeuronCluster('n3', 10, **neuron_params))

syn1 = InputSynapseCluster(net['n1'], INPUT_CURRENT, INPUT_SD)

syn2 = SynapseCluster(net['n1'], net['n2'], **synapse_params)
syn3 = SynapseCluster(net['n2'], net['n3'], **synapse_params)
syn4 = SynapseCluster(net['n3'], net['n1'], **synapse_params)

if __name__ == '__main__':
    print(net)
    STEPS = int((TIME_FINAL - TIME_START)/STEP_SIZE)
    ts = (np.arange(STEPS)-TIME_START) * STEP_SIZE
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
        n1_list.append(net.neurons['n1'].V)
        n2_list.append(net.neurons['n2'].V)
        n3_list.append(net.neurons['n3'].V)
        s1_list.append(np.average(syn1.current()))
        s2_list.append(np.average(syn2.current()))
        s3_list.append(np.average(syn3.current()))
        s4_list.append(np.average(syn4.current()))
        net.update(STEP_SIZE)

    n1_arr = np.array(n1_list).T
    n2_arr = np.array(n2_list).T
    n3_arr = np.array(n3_list).T

    fig, axes = plt.subplots(3, 1, figsize=(15, 7))
    for n1_seq in n1_arr:
        axes[0].plot(ts, n1_seq, 'b-', alpha=.1)
    for n2_seq in n2_arr:
        axes[0].plot(ts, n2_seq, 'g-', alpha=.1)
    for n3_seq in n3_arr:
        axes[0].plot(ts, n3_seq, 'm-', alpha=.1)
    axes[0].plot(ts, np.average(n1_arr, axis=0), 'b--', label='$n_1$')
    axes[0].plot(ts, np.average(n2_arr, axis=0), 'g--', label='$n_2$')
    axes[0].plot(ts, np.average(n3_arr, axis=0), 'm--', label='$n_3$')

    axes[0].set_ylabel('mV')
    axes[1].plot(ts, s1_list, 'k:', label=r'$\mapsto n1$')
    axes[1].plot(ts, s2_list, 'g-', label=r'$n1 \to n2$')
    axes[1].plot(ts, s3_list, 'g-', label=r'$n2 \to n3$')
    axes[1].plot(ts, s4_list, 'm-', label=r'$n3 \to n1$')
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
