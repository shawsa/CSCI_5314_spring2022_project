
from .neuron import NeuronCluster, InputNeuronCluster
from .synapse import SynapseCluster

class Network:
    def __init__(self, start_time: float, dt: float, num_steps: int):
        self.start_time = start_time
        self.dt = dt
        self.num_steps = num_steps

        self.time = None
        self.time_index = None

        self.neurons = {}
        self.synapses = {}

    def add_neurons(self, *args):
        for arg in args:
            assert isinstance(arg, NeuronCluster)
            assert arg.name not in self.neurons.keys()
            self.neurons[arg.name] = arg

    def add_synapse(self,
                    pre: str,
                    post: str,
                    syn: SynapseCluster):

        assert pre in self.neurons.keys()
        assert post in self.neurons.keys()
        name = (pre, post)
        assert name not in self.synapses.keys()

        # if the presynaptic neuron already has one of this
        # type, we'll just reuse it.
        for syn2 in self[pre].outputs:
            if syn == syn2:
                self[post].inputs.append(syn2)
                self.synapses[name] = syn2
                return

        self[pre].outputs.append(syn)
        self[post].inputs.append(syn)
        self.synapses[name] = syn

        syn.pre_size = self[pre].size

    def reset(self):
        self.time = self.start_time
        self.time_index = 0
        for neuron in self.neurons.values():
            if isinstance(neuron, InputNeuronCluster):
                neuron.set_sim_params(self.start_time, self.dt)
            neuron.reset()

    def update(self):
        for neuron in self.neurons.values():
            neuron.compute_update(self.time_index, self.dt)

        for neuron in self.neurons.values():
            neuron.store_update()

        self.time_index += 1
        self.time = self.start_time + self.time_index * self.dt

    def simulate(self):
        self.reset()
        yield self
        for _ in range(self.num_steps):
            self.update()
            yield self

    def __getitem__(self, key):
        '''Dictionary like access of neurons and synapses.'''
        if isinstance(key, tuple):
            # assume synaptic connection
            return self.synapses[key]
        elif key in self.neurons.keys():
            return self.neurons[key]
        else:
            raise ValueError(f'Cannot locate {key} in {self.__repr__()}.')

    def nodes(self):
        return self.neurons.keys()

    def edges(self):
        return self.synapses.keys()

    def __str__(self):
        return f'Network: {len(self.neurons)} neurons, ' + \
               f'{sum(map(lambda x: 1, self.edges()))} synapses'

    def __repr__(self):
        return str(self) + f' @ {id(self)}'

    def print_full(self):
        print(str(self))
        print('\tNeurons:')
        for neuron in self.neurons.values():
            print(f'\t\t{str(neuron)}')
        print('\tSynapses:')
        for (pre, post), syn in self.synapses.items():
            print(f'\t\t {pre}->{post}: {str(type(syn))}')
