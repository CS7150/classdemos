import torch
from matplotlib import pyplot as plt
from baukit import PlotWidget, show

class Sign(torch.nn.Module):
    '''
    The Sign nonlinearity is a step function that returns +1 for all positive
    numbers and -1 for all negative numbers.  Zero stays as zero.
    '''
    def forward(self, x):
        return x.sign()

class McCulloughPittsNeuron(torch.nn.Module):
    '''
    A McCullough-Pitts Neuron.  It computes a weighted sum of any number of inputs,
    then it thresholds the output through a nonlinear activation step function.
    It pulls named inputs from an input dictionary and puts output into the
    dictionary.  That allows networks to be created by sequencing neurons and
    connecting them by using dictionary names.

    Examples:

        net = McCulloughPittsNeuron(
                weight_a = 0.5,
                weight_b = -0.3,
                weight_c = 2.0,
                bias     = 1.0)
        print(net(dict(
                a=Tensor([1.0]),
                b=Tensor([-1.0]),
                c=Tensor([-1.0])))['out'])

    The above creates a single neuron with three inputs a, b, and c plus some bias.
    It is invoked by providing a dictionary of all the inputs as tensors.

        net = torch.nn.Sequential(
            McCulloughPittsNeuron(weight_a=-1.0, weight_b=1.0, output_name='d'),
            McCulloughPittsNeuron(weight_b=1.0, weight_d=1.0, bias=1.0),
        )
        print(net(dict(a=Tensor([1.0]), b=Tensor([-1.0])))['out'])

    The above creates and runs a network of two neurons in this configuration:
    ```
             a -----> +----------+
                      | Neuron 0 | ---> d --+
             b ---+-> +----------+          +--> +----------+
                  |                              | Neuron 1 | ---> out
                  +----------------------------> +----------+
    ```
    As the sequence is run, the dictionary grows; after the first neuron is run,
    the dictionary contains a, b, and d.  After the second neuron is ru , the
    final dictionary contains a, b, d, and out.
    '''
    def __init__(self, bias=0.0, activation=Sign, output_name='out', **kwargs):
        '''
        Construct a neuron by specifying any number of input weights in the arguments:
        
            weight_a:    The weight for the 'a' input.
                         Each `weight_x` in the constructor adds an input named 'x'.
            bias:        The constant bias to add to the weighted sum.
            output_name: The output name, defaults to 'out'.
            activation:  The nonlinearity to use; defaults to the "Sign" step function.
        '''
        super().__init__()
        
        # We use the pytorch Linear module with a one-dimenaional output
        self.summation = torch.nn.Linear(len(kwargs), 1)
        self.activation = None if activation is None else activation()
        self.output_name = output_name
        self.input_names = []
        with torch.no_grad():
            self.summation.bias[...] = bias
            for k, v in kwargs.items():
                assert k.startswith('weight_'), f'Bad argument {k}'
                self.summation.weight[0, len(self.input_names)] = v
                self.input_names.append(k[7:])

    def forward(self, inputs):
        '''
        The inputs should be a dictionary containing the expected input keys.
        The results are computed.  Then the return value will be a copy of the
        input dictionary, with the additional output tensor added.
        '''
        if isinstance(inputs, torch.Tensor):
            # If the inputs are just a tensor, then split it into a dictionary.
            inputs = {k: inputs[:,i] for i, k in enumerate(self.input_names)}
        state = inputs.copy()
        assert self.output_name not in state, f'Multiple {self.output_name}\'s conflict'
        x = torch.stack([inputs[v] for v in self.input_names], dim=1)
        x = self.summation(x)[:,0]
        if self.activation is not None:
            x = self.activation(x)
        state[self.output_name] = x
        return state
    
    def extra_repr(self):
        return f'input_names={self.input_names}, output_name=\'{self.output_name}\''

def visualize_logic(nets, arg1='a', arg2='b'):
    '''
    Pass any number of McCullough-Pitts neurons or neural networks with two
    inputs named 'a' and 'b', and it will visualize all of their logic, using
    white squares to indicate +1, black squares to indicate -1, and orange
    squares to indicate intermediate values.
    '''
    grid = torch.Tensor([[
        [-1.0, 1.0],
        [-1.0, 1.0],
    ], [
        [ 1.0, 1.0],
        [-1.0,-1.0],
    ]])
    a, b = grid
    def make_viz(n, case=()):
        if isinstance(n, list):
            return [make_viz(net, case + (str(i+1),)) for i, net in enumerate(n)]
        def make_plot(fig):
            with torch.no_grad():
                out = n({arg1: a.view(-1), arg2: b.view(-1)})['out'].view(a.shape)
            [ax] = fig.axes
            ax.imshow(out, cmap='hot', extent=[-2,2,-2,2], vmin=-1, vmax=1)
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            ax.tick_params(length=0)
            ax.set_xticks([-1, 1], [f'{arg1}=-1', f'{arg1}=1'])
            ax.set_yticks([-1, 1], [f'{arg2}=-1', f'{arg2}=1'])
        return [PlotWidget(make_plot, figsize=(1.1,1.1), dpi=100, bbox_inches='tight'),
                show.style(margin='0 0 20px 45%', textAlign='right'), f'case {" ".join(case)}']
    show(show.WRAP, make_viz(nets))

# Abbreviation
MPNeuron = McCulloughPittsNeuron

class Input(torch.nn.Module):
    '''
    Returns the 'out' tensor from the dictionary.
    '''
    def __init__(self, input_names):
        super().__init__()
        self.input_names = [name for name in input_names]
    def forward(self, x):
        return {k: x[:,i] for i, k in enumerate(self.input_names)}
    def extra_repr(self):
        return f'input_names=\'{self.input_names}\''

class Inplace(torch.nn.Module):
    '''
    Invokes the given function on a single key in the dictionary.
    '''
    def __init__(self, input_name, f):
        super().__init__()
        self.input_name = input_name
        self.f = f
    def forward(self, x):
        output = x.copy()
        output[self.input_name] = f(x[self.input_names])
    def extra_repr(self):
        return f'input_name=\'{self.input_name}\, f={self.f}'

class Output(torch.nn.Module):
    '''
    Returns the 'out' tensor from the dictionary.
    '''
    def __init__(self, output_name='out'):
        super().__init__()
        self.output_name = output_name
    def forward(self, x):
        return x[self.output_name]
    def extra_repr(self):
        return f'output_name=\'{self.output_name}\''

