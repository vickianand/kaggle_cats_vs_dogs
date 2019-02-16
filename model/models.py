import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoConvOnePool(nn.Module):
    """Repetetive network structure found in VGG net
    Spatial-dimension is reduced by 2:
        Conv layers don't change the spatial dimension (kernel=3, padding=1)
        MaxPool layer halves the spatial dimension
    Depth dimensions should be passed as argument
    """

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # stride=2 (default value is same as kernel)
        )

    def forward(self, x):
        return self.net(x)


class VggTypeNet(nn.Module):
    """
    Inspired from VGG Net structure
    """

    def __init__(self, width=64, num_classes=1, channel_list=[64, 128, 64]):
        """
        default values set to those for cat vs dog classification
        assuming in_channel = 3
        """
        super().__init__()

        net_list = []
        inp_dim = 3
        for out_dim in channel_list:
            net_list.append(TwoConvOnePool(inp_dim, out_dim))
            inp_dim = out_dim

        self.conv_net = nn.Sequential(*net_list)
        # spatial dimension reduces by a factor of 2**len(channel_list)

        self.fc_in_features = channel_list[-1] * (width // 2 ** len(channel_list)) ** 2

        self.f_c_layer = nn.Linear(
            in_features=self.fc_in_features, out_features=num_classes
        )

    def forward(self, x):
        return self.f_c_layer(self.conv_net(x).reshape(-1, self.fc_in_features))


'''
class SwechhaNet(nn.Module):
    """
    Structure taken from our submission for Kaggle competition for IFT6390
    This was implemented by Swechha; so the name
    """

    def __init__(self, width, num_classes):
        super().__init__()
        self.layer1 = nn.Conv2d(3, width // 4, 3, stride=2)
        self.relu = nn.ReLU()
        self.layer2 = nn.Conv2d(width // 4, width // 2, 3, stride=2)
        self.layer3 = nn.Conv2d(width // 2, width // 2, 3, stride=2, padding=1)
        self.layer4 = nn.Conv2d(width // 2, width, 3, stride=2, padding=1)
        self.layer5 = nn.Conv2d(width, 2 * width, 12, stride=1)
        self.layer6 = nn.Conv2d(2 * width, 2 * width, 1)
        self.layer7 = nn.Conv2d(2 * width, num_classes, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out1 = self.layer1(x)
        input2 = self.relu(out1)
        out2 = self.layer2(input2)
        input3 = self.relu(out2)
        out3 = self.layer3(input3)
        input4 = self.relu(out3)
        out4 = self.layer4(input3)
        input5 = self.relu(out4)
        out5 = self.layer5(input5)
        input6 = self.relu(out5)
        out6 = self.layer6(input6)
        input7 = self.relu(out6)
        out7 = self.layer7(input7)
        return out7.squeeze()
'''
