import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fruit_fly import (
    Network,
    NeuronCluster,
    DEFAULT_NEURON_PARAMS,
    REIP_PARAMS,
    SynapseCluster,
    NMDA,
    NMDA_PARAMS,
    GABAA_PARAMS,
    InputSynapseCluster
)

INPUT_CURRENT = -200
INPUT_SD = .05

TIME_START = 0.0
TIME_FINAL = 0.2
STEP_SIZE = 1e-3

net = Network()

NUM_CLUSTERS = 15

excitatory_neurons = [f'R{num}' for num in range(NUM_CLUSTERS)]

for name in excitatory_neurons:
    net.add(NeuronCluster(name, 10, **DEFAULT_NEURON_PARAMS))

# exitatory ring
for n1, n2 in zip(excitatory_neurons[:-1], excitatory_neurons[1:]):
    NMDA(net[n1], net[n2], max_conductance=1, **NMDA_PARAMS)
    NMDA(net[n2], net[n1], max_conductance=1, **NMDA_PARAMS)

NMDA(net[f'R{NUM_CLUSTERS-1}'], net['R0'], max_conductance=1, **NMDA_PARAMS)
NMDA(net['R0'], net[f'R{NUM_CLUSTERS-1}'], max_conductance=1, **NMDA_PARAMS)

stim0 = InputSynapseCluster(net['R0'], INPUT_CURRENT, INPUT_SD)
stim5 = InputSynapseCluster(net[f'R{NUM_CLUSTERS//2}'], 0, INPUT_SD)

# add inhibition
net.add(NeuronCluster('I', 10, **REIP_PARAMS))
SynapseCluster(net['I'], net['I'], max_conductance=-1, **GABAA_PARAMS)
for name in excitatory_neurons:
    SynapseCluster(net['I'], net[name], max_conductance=-1, **GABAA_PARAMS)
    NMDA(net[name], net['I'], max_conductance=1, **NMDA_PARAMS)


if __name__ == '__main__':
    print(net)
    STEPS = int((TIME_FINAL - TIME_START)/STEP_SIZE)
    ts = (np.arange(STEPS)-TIME_START) * STEP_SIZE
 
    neuron_plot_dict = {name: [] for name in excitatory_neurons}
    syn_plot_dict = {name: [] for name in excitatory_neurons}
    inhibitory_plot = []
    for t in tqdm(ts):
        if t > 0.1:
            stim0.output_current = 0
        net.update(STEP_SIZE)
        for name in excitatory_neurons:
            neuron_plot_dict[name].append(np.average(net[name].V))
            syn_plot_dict[name].append(
                    np.average(net[name].axonal_connections[0].current()))
        inhibitory_plot.append(np.average(net['I'].V))

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    for name in excitatory_neurons:
        axes[0].plot(ts, neuron_plot_dict[name])
        axes[1].plot(ts, syn_plot_dict[name])
    axes[0].plot(ts, inhibitory_plot, 'k-')

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
