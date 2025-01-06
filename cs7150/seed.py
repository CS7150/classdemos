import torch, numpy

def seeded_net(seed, net, uniform=False):
    '''
    Deterministically seeds the given network with pseudorandom weights
    determined by the given seed, to make classroom exercises more
    reproducible and satsifying.  Example:

       `net = seeded_net(1, MyNetwork())`

    Note: this function uses pseudorandom numbers in the range [-1, 1]
    without any clever Xavier or Kaiming scaling.
    '''
    prng = numpy.random.RandomState(seed)
    with torch.no_grad():
        for p in net.parameters():
            if uniform:
                p[...] = torch.tensor(prng.uniform(-1.0, 1.0, p.numel())).reshape(p.shape)
            else:
                p[...] = torch.tensor(prng.randn(p.numel())).reshape(p.shape)
    return net

def non_linearly_separable_data(seed=1, n=100, integer_labels=False):
    '''
    Returns a standard set of n (100) 2-dimensional data points,
    along with labels that are not linearly separable. Example:

        `data, labels = non_linearly_separable_data()`
    '''
    prng = numpy.random.RandomState(seed)
    data = torch.Tensor(prng.randn(n, 2))
    labels = torch.Tensor(numpy.stack([
        (d[0].sign() == d[1].sign())
        for d in data
    ]))
    if integer_labels:
        # Return {0, 1} integer labels
        labels = labels.long()
    else:
        # Return +- 1 float labels.
        labels = labels * 2 - 1
    return data, labels
