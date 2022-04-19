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
    ACETYLCHOLINE_PARAMS
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


# Big problems with parameters
# I think there might be some kind of unit errors


# times in ms
TIME_START = 0.0
TIME_FINAL = 0.1  # adjusted to match step size
STEP_SIZE = 1e-7

steps = int((TIME_FINAL - TIME_START)/STEP_SIZE)
plot_frames = 100
plot_stride = steps//plot_frames

ts = (np.arange(steps)-TIME_START) * STEP_SIZE

net = Network(TIME_START, STEP_SIZE)

net.add(
    InputNeuron('input', 10, 50, (0.0, 0.05)),
    NeuronCluster('n1', 10, **DEFAULT_NEURON_PARAMS)
)
syn = SynapseCluster(
        net['input'],
        net['n1'],
        max_conductance=2.1,
        **ACETYLCHOLINE_PARAMS)

net.reset()

if __name__ == '__main__':
    net.print_full()
    print(f'{steps} steps at size={STEP_SIZE}')

    ts = [net.time]
    n1_voltages = [np.average(net['n1'].V)]
    syn_currents = [np.average(syn.current)]

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
            'k-',
            label=syn.name)

    current_axis.set_ylim(-2_000, 0)
    current_axis.set_ylabel(r'$\mu A$')

    # spike times

    input_spike_times = []
    input_spike_height = 0
    input_spike_line, = spike_axis.plot(
            input_spike_times,
            [input_spike_height]*len(input_spike_times),
            'k.')
    n1_spike_times = []
    n1_spike_height = 1 
    n1_spike_line, = spike_axis.plot(n1_spike_times,
                                     [n1_spike_height]*len(n1_spike_times),
                                     'b.',
                                     label='n1')

    spike_axis.set_ylabel('spikes')

    # gating variables

    input_gating_values = [np.average(net[('input', 'n1')].gating)]
    gating_axis.set_ylim(-1, 5)
    input_gating_line, = gating_axis.plot(ts, input_gating_values,
                                          'k-')
    gating_axis.set_ylabel('gating variable')

    axes[-1].set_xlim(TIME_START, TIME_FINAL)
    axes[-1].set_xlabel('t (s)')

    print(net['input'].next_spike_index)

    for step in tqdm(range(steps)):
        net.update()
        # log data
        ts.append(net.time)
        n1_voltages.append(np.average(net['n1'].V))
        syn_currents.append(np.average(syn.current))
        input_gating_values.append(np.average(net[('input', 'n1')].gating))
        if np.any(net['input'].firing):
            input_spike_times.append(net.time)
        if np.any(net['n1'].firing):
            n1_spike_times.append(net.time)

        # print if step is plot frame
        if step % plot_stride == 0 or step == steps-1:
            n1_voltage_line.set_data(ts, n1_voltages)
            input_spike_line.set_data(
                    input_spike_times,
                    [input_spike_height]*len(input_spike_times))
            n1_spike_line.set_data(
                    n1_spike_times,
                    [n1_spike_height]*len(n1_spike_times))
            syn_current_line.set_data(ts, syn_currents)
            input_gating_line.set_data(ts, input_gating_values)
            fig.canvas.draw()
        plt.pause(STEP_SIZE)

    print('Press enter when finished with plot.')
    input()

