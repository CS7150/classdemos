import torch, numpy
from collections import OrderedDict
from baukit import Widget, Menu, show, PlotWidget

class LossSurfaceWidget(Widget):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        self.param_map = OrderedDict([(f'{name}{index}', (param, index))
            for name, param in mlp.named_parameters()
            for index in numpy.ndindex(param.shape)])
        menu1 = Menu(param_map.keys())
        plot1 = PlotWidget(self.draw_loss_curve, mlp=mlp,
                param_name=menu1.prop('value'), figsize=(3,2.8))
        menu2 = Menu(param_map.keys())
        plot2 = PlotWidget(self.draw_loss_curve, mlp=mlp,
                param_name=menu2.prop('value'), figsize=(3,2.8))
        plot3 = PlotWidget(self.draw_loss_surface, mlp=mlp,
                   param_name_1=menu1.prop('value'),
                   param_name_2=menu2.prop('value'), figsize=(3,3))
        self.content = [[[menu1, plot1], [menu2, plot2], plot3]]

    def _repr_html_(self):
        return show.html(show.TIGHT, self.content)

    def draw_loss_curve(self, fig, mlp=None, param_name=None, lim=10):
        if mlp is None or param_name not in self.param_map:
            return
        [ax] = fig.axes
        ax.clear()
        param, index = self.param_map[param_name]
        offsets = torch.linspace(-lim, lim, 101)
        losses = []
        with torch.no_grad():
            original_w = param[index].detach().item()
            for offset in offsets:
                param[index] = original_w + offset
                preds = mlp(data)[:,0]
                loss = ((preds - labels) ** 2).mean()
                losses.append(loss.item())
            param[index] = original_w
        ax.set_title(param_name)
        ax.plot(offsets, losses)

    def draw_loss_surface(self, fig, mlp=None, param_name_1=None, param_name_2=None, lim=10):
        if mlp is None or param_name_1 not in param_map or param_name_2 not in param_map:
            return
        [ax] = fig.axes
        ax.clear()
        offsets = torch.linspace(-lim, lim, 101)
        param1, index1 = self.param_map[param_name_1]
        param2, index2 = self.param_map[param_name_2]
        losses = torch.zeros(101, 101)
        with torch.no_grad():
            original_1 = param1[index1].detach().item()
            original_2 = param2[index2].detach().item()
            for i in range(101):
                param1[index1] = offsets[i]
                for j in range(101):
                    param2[index2] = offsets[j]
                    preds = mlp(data)[:,0]
                    loss = ((preds - labels) ** 2).mean()
                    losses[i,j] = loss
        ax.imshow(losses, cmap='hot', extent=[-lim,lim,-lim,lim])
