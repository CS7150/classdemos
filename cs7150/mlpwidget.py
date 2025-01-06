import torch, numpy
from baukit import PlotWidget, show, Widget, Range, Numberbox

class MLPHistoryWidget(Widget):
    '''
    A widget to show what is going on inside an MLP network as it trains.
    It assumes that the network operates as a binary classifier with +/1 outputs,
    classifying 2-dimensional inputs in the range [-3, 3].  Example:

        `show(MLPHistoryWidget(data=D, labels=L, history=[net1, net2, net3]))`
    '''
    def __init__(self, data=None, labels=None, history=None, maxWidth=1000):
        super().__init__()
        # We visualize the data points in the label classes, as well as
        # a history of trained networks.
        self.history = history if history is not None else []
        self.data = data if data is not None else []
        self.labels = labels if labels is not None else []
        self.maxWidth = maxWidth
        # The layout consists of a plot with a scrubber to show the iteration count
        self.plot = PlotWidget(
                self.visualize_net,
                index=max(0, len(self.history) - 1),
                ncols=3,
                nrows=2,
                figsize=(11,5.5),
                bbox_inches='tight',
                gridspec_kw={'hspace': 0.25, 'height_ratios': [2,1]})
        scrubber = Range(min=0, max=len(self.history), value=self.plot.prop('index'))
        numbox = Numberbox(value=self.plot.prop('index'))
        self.content = [
            [
                [show.style(alignContent='center'), 'Iteration'],
                numbox,
                show.style(flex=20), scrubber
            ],
            self.plot
        ]
        # If you click on the history, it also moves the scrubber.
        self.plot.on('click', self.plot_click)

    def _repr_html_(self):
        return show.html(show.style(maxWidth=self.maxWidth), self.content)

    def visualize_net(self, fig, index=0):
        '''
        The plot rendering method for the widget.
        '''
        fig.subplots_adjust(0.02, 0.02, 0.98, 0.98)
        def endpoints(w, b, scale=10):
            if abs(w[1]) > abs(w[0]):
                x0 = torch.tensor([-scale, scale]).to(w.device)
                x1 = (-b - w[0] * x0) / w[1]
            else:
                x1 = torch.tensor([-scale, scale]).to(w.device)
                x0 = (-b - w[1] * x1) / w[0]
            return torch.stack([x0, x1], dim=1)

        if len(fig.axes) > 4:
            gs = fig.axes[3].get_gridspec()
            for ax in fig.axes[3:]:
                ax.remove()
            fig.add_subplot(gs[1, :])
        ax1, ax2, ax3, ax4 = fig.axes[:4]
        ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()
        if index >= len(self.history):
            return

        # Visualize the network output
        net, data = self.history[index]
        grid = torch.stack([
            torch.linspace(-3, 3, 200)[None, :].expand(200, 200),
            torch.linspace(3, -3, 200)[:, None].expand(200, 200),
        ])
        ax1.set_title('network output')
        with torch.no_grad():
            score = net(grid.permute(1, 2, 0).reshape(-1, 2))
            if score.shape[1] == 2: # Allow 2-d logit output or 1-d raw output
                score = score.softmax(1) * 2 - 1
            score = score[:,0].reshape(200, 200).detach()

        ax1.imshow(score, cmap='hot', extent=[-3,3,-3,3], vmin=-1, vmax=1)
        ax2.imshow(score, cmap='hot', extent=[-3,3,-3,3], alpha=0.2, vmin=-1, vmax=1)

        # Visualize the training data
        ax2.set_title('training data')
        ax2.set_ylim(-3, 3)
        ax2.set_xlim(-3, 3)
        ax2.set_aspect(1.0)
        ax2.scatter([d[0] for d, l in zip(self.data, self.labels) if l > 0],
                    [d[1] for d, l in zip(self.data, self.labels) if l > 0])
        ax2.scatter([d[0] for d, l in zip(self.data, self.labels) if l <= 0],
                    [d[1] for d, l in zip(self.data, self.labels) if l <= 0])

        # Visualize the geometry of the first-layer folds.
        ax3.set_title('first layer folds')
        module = [m for m in net.modules() if isinstance(m, torch.nn.Linear)][0]
        w = module.weight.detach().cpu()
        b = module.bias.detach().cpu()
        e = torch.stack([endpoints(wc, bc) for wc, bc in zip(w, b)])
        for ep in e:
            ax3.plot(ep[:,0], ep[:,1], '#00aa00', linewidth=0.75, alpha=0.33)
        ax3.set_ylim(-3, 3)
        ax3.set_xlim(-3, 3)
        ax3.set_aspect(1.0)

        # Draw the loss curve and any other plotted stats.
        ax4.set_title('training curve')
        ax4.set_xlabel('iteration')
        ax4.set_yscale('log')
        ax4.axvline(index, color='red', linewidth=0.5)
        for k, v in data.items():
            label = f'{k} = {v:.3g}'
            ax4.plot(range(len(self.history)),
                     [h[1][k] for h in self.history],
                     linewidth=0.5, label=label)
        ax4.legend()

    def plot_click(self, e):
        '''
        Handles click events for the widget, moves the scrubber to the
        iteration corresponding to the user click.
        '''
        loc = self.plot.event_location(e)
        if loc.axis == 3: # Fourth axis
            self.plot.index = max(0, min(len(self.history) - 1, int(loc.x + 0.5)))
            self.plot.redraw()

