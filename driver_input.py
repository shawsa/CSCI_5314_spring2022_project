#!/usr/bin/python3
'''
This is to test the input neurons as described in
Spatial Orientation Task subsection in the Methods
section of Su et al. 2017.

It consists of ...
'''
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
    ACETYLCHOLINE_PARAMS
)

# times in s
TIME_START = 0.0
STEP_SIZE = 1e-7
TIME_FINAL = 0.1  # adjusted to match step size
plot_frames = 100


steps = int((TIME_FINAL - TIME_START)/STEP_SIZE)
plot_stride = steps//plot_frames

net = Network(TIME_START, STEP_SIZE, steps)

net.add_neurons(
    InputNeuronCluster('input', 10, 50, (0.0, 0.05)),
    NeuronCluster('n1', 10, **DEFAULT_NEURON_PARAMS)
)

net.add_synapse(
    'input',
    'n1',
    SynapseCluster(
        max_conductance=2.1,
        **ACETYLCHOLINE_PARAMS))

if __name__ == '__main__':
    net.print_full()
    print(f'{steps} steps at size={STEP_SIZE}')

    ts = [net.time]
    n1_voltages = [net['n1'].V]
    syn_currents = [net[('input', 'n1')].current(net['n1'].V)]

    plt.ion()
    fig, axes = plt.subplots(4, 1, figsize=(15, 15), sharex=True)

    spike_axis = axes[0]
    voltage_axis = axes[1]
    current_axis = axes[2]
    gating_axis = axes[3]

    # voltage plot

    n1_voltage_line, = voltage_axis.plot(ts, n1_voltages, 'b-', label='$n_1$')
    voltage_axis.set_ylim(-80, -40)
    voltage_axis.set_ylabel('mV')

    # current plot

    syn_current_line, = current_axis.plot(
            ts,
            syn_currents,
            'k-')

    current_axis.set_ylim(-2_000, 0)
    current_axis.set_ylabel('pA')

    # spike times

    input_spike_height = 0
    input_spike_line, = spike_axis.plot([], [], 'k.', label='input')
    n1_spike_height = 1 
    n1_spike_line, = spike_axis.plot([], [], 'b.', label='n1')

    spike_axis.set_ylabel('spikes')
    spike_axis.set_ylim(-5, 5)
    spike_axis.legend()

    # gating variables

    input_gating_values = [np.average(net[('input', 'n1')].gating)]
    gating_axis.set_ylim(-1, 5)
    input_gating_line, = gating_axis.plot(ts, input_gating_values,
                                          'k-')
    gating_axis.set_ylabel('gating variable')

    axes[-1].set_xlim(TIME_START, TIME_FINAL)
    axes[-1].set_xlabel('t (s)')

    net.reset()
    for step in tqdm(range(steps)):
        net.update()

        # log data
        ts.append(net.time)
        n1_voltages.append(np.average(net['n1'].V))
        syn_currents.append(net[('input', 'n1')].current(net['n1'].V))
        input_gating_values.append(net[('input', 'n1')].gating)

        # print if step is plot frame
        if step % plot_stride == 0 or step == steps-1:
            n1_voltage_line.set_data(ts, n1_voltages)
            spike_times = [ts[i] for i in net['input'].firing_time_indices]
            input_spike_line.set_data(
                    spike_times,
                    [input_spike_height]*len(spike_times))
            spike_times = [ts[i] for i in net['n1'].firing_time_indices]
            n1_spike_line.set_data(
                    spike_times,
                    [n1_spike_height]*len(spike_times))
            syn_current_line.set_data(ts, syn_currents)
            input_gating_line.set_data(ts, input_gating_values)
            fig.canvas.draw()
        plt.pause(STEP_SIZE)

    print('Press enter when finished with plot.')
    input()

