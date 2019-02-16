import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBatchNorm(nn.Module):
    """ Implements 2d batch-norm that does not track the running mean and variance
        and always uses batch statistics in both training and eval modes.
    """

    def __init__(self, num_channels):
        super().__init__()
        # self.num_channels = num_channels
        # register and initialize the weight and bias
        self.w = nn.Parameter(1 + 0.1 * torch.randn(num_channels))
        self.b = nn.Parameter(0 + 0.1 * torch.randn(num_channels))

    def forward(self, x):
        """
        Arguments:
            X : shape (B, C, H, W); where C == num_channels
        """
        mu = x.mean(dim=0)  # shape (C, H, W)
        var = x.var(dim=0)  # shape (C, H, W)

        x = x - mu
        x = x / (var ** 0.5)  # shape (B, C, H, W)

        # need to broadcast w and b of shapes (C,) to x of shape (B, C, H, W)
        return x * self.w.unsqueeze(1).unsqueeze(2) + self.b.unsqueeze(1).unsqueeze(2)


class TwoConvOnePool(nn.Module):
    """ Repetetive network structure found in VGG net
        Spatial-dimension is reduced by 2:
        Conv layers don't change the spatial dimension (kernel=3, padding=1)
        MaxPool layer halves the spatial dimension
        Depth dimensions should be passed as argument
    """

    def __init__(self, in_channels=3, out_channels=64, batch_norm=False):
        super().__init__()
        self.net = None
        if batch_norm:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                SimpleBatchNorm(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                SimpleBatchNorm(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

    def forward(self, x):
        return self.net(x)


class VggTypeNet(nn.Module):
    """ Inspired from VGG Net structure
    """

    def __init__(
        self, width=64, num_classes=1, channel_list=[64, 128, 64], batch_norm=False
    ):
        """ default values set to those for cat vs dog classification
            assuming in_channel = 3
        """
        super().__init__()

        net_list = []
        inp_dim = 3
        for out_dim in channel_list:
            net_list.append(TwoConvOnePool(inp_dim, out_dim, batch_norm=batch_norm))
            inp_dim = out_dim

        self.conv_net = nn.Sequential(*net_list)
        # spatial dimension reduces by a factor of 2**len(channel_list)

        self.fc_in_features = channel_list[-1] * (width // 2 ** len(channel_list)) ** 2

        self.f_c_layer = nn.Linear(
            in_features=self.fc_in_features, out_features=num_classes
        )

    def forward(self, x):
        return self.f_c_layer(self.conv_net(x).reshape(-1, self.fc_in_features))

