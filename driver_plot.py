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
    InputSynapseCluster,
    Probe,
    FiringRateProbe,
    neuron_ave_voltage,
    syn_ave_current,
    syn_ave_gating
)

INPUT_CURRENT = -200
INPUT_SD = .05
CUTOFF_TIME = 1.0

TIME_START = 0.0
TIME_FINAL = 3.0
STEP_SIZE = 1e-3

FRAMES = 100

# synaptic time scale is too slow to see change
# lower it for plotting purposes
NMDA_PARAMS['time_constant'] = 1
FIRING_HALF_LIFE = .5

STEPS = int((TIME_FINAL - TIME_START)/STEP_SIZE)
firing_decay = FIRING_HALF_LIFE / np.log(2)
plot_stride = STEPS//FRAMES
ts = (np.arange(STEPS)-TIME_START) * STEP_SIZE

net = Network()

net.add(
        NeuronCluster('n1', 10, **DEFAULT_NEURON_PARAMS),
        NeuronCluster('n2', 10, **DEFAULT_NEURON_PARAMS),
        NeuronCluster('n3', 10, **DEFAULT_NEURON_PARAMS),
        )

stim0 = InputSynapseCluster(net['n1'], INPUT_CURRENT, INPUT_SD)
syns = [
        NMDA(net['n1'], net['n2'], max_conductance=1, **NMDA_PARAMS),
        NMDA(net['n2'], net['n3'], max_conductance=1, **NMDA_PARAMS)
        ]

plt.ion()
fig, axes = plt.subplots(4, 1, figsize=(10, 15))
for ax in axes:
    ax.set_xlabel('t (ms)')
    ax.set_xlim(TIME_START, TIME_FINAL)

axes[0].set_ylim(-80, -40)
axes[0].set_ylabel('mV')

axes[1].set_ylim(-1, 100)
axes[1].set_ylabel('Hz')

axes[2].set_ylim(-250, 20)
axes[2].set_ylabel('mA')

axes[3].set_ylim(-.1, 1.1)
axes[3].set_ylabel('gating')

net.add(Probe(
            net['n1'],
            *axes[0].plot([], [], 'b-', label='$n_1$'),
            neuron_ave_voltage),
        Probe(
            net['n2'],
            *axes[0].plot([], [], 'g-', label='$n_2$'),
            neuron_ave_voltage),
        Probe(
            net['n3'],
            *axes[0].plot([], [], 'm-', label='$n_3$'),
            neuron_ave_voltage)
        )

net.add(FiringRateProbe(
            net['n1'],
            *axes[1].plot([], [], 'b-', label='$n_1$'),
            firing_decay),
        FiringRateProbe(
            net['n2'],
            *axes[1].plot([], [], 'g-', label='$n_2$'),
            firing_decay),
        FiringRateProbe(
            net['n3'],
            *axes[1].plot([], [], 'm-', label='$n_3$'),
            firing_decay)
        )

net.add(Probe(stim0,
              *axes[2].plot([], [], 'k:', label=r'$\mapsto n_1$'),
              syn_ave_current),
        Probe(net[('n1', 'n2')],
              *axes[2].plot([], [], 'b-', label=r'$n_1 \to n_2$'),
              syn_ave_current),
        Probe(net[('n2', 'n3')],
              *axes[2].plot([], [], 'g-', label=r'$n_2 \to n_3$'),
              syn_ave_current)
        )

net.add(Probe(net[('n1', 'n2')],
              *axes[3].plot([], [], 'b-', label=r'$n_1 \to n_2$'),
              syn_ave_gating),
        Probe(net[('n2', 'n3')],
              *axes[3].plot([], [], 'g-', label=r'$n_2 \to n_3$'),
              syn_ave_gating)
        )

for ax in axes:
    ax.legend()

plt.show()

if __name__ == '__main__':
    print(net)
    for i, t in enumerate(tqdm(ts)):
        if t > CUTOFF_TIME:
            stim0.output_current = 0
        net.update(STEP_SIZE)
        if i % plot_stride == 0:
            net.update_plot()
            fig.canvas.draw()
            plt.pause(1e-1)
    net.update_plot()
    fig.canvas.draw()
    plt.pause(1e-1)
    print('Press enter to close')
    input()

