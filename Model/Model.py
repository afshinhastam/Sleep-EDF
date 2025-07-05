import torch
import torch.nn as nn

import torch
import torch.nn as nn

# TimeDistributed wrapper for applying a module over time steps (batch, time, ...)
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        b, t = x.size(0), x.size(1)
        x_reshaped = x.contiguous().view(b * t, *x.size()[2:])
        y = self.module(x_reshaped)
        y = y.contiguous().view(b, t, *y.size()[1:])
        return y


class CNNBlock(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, 3, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(16, 32, 3, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),

            nn.Conv1d(32, 64, 3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),

            nn.Conv1d(64, 128, 3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.05),

            nn.Conv1d(128, 256, 3, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.features(x)


class FeatureAttention(nn.Module):
    def __init__(self, embed_dim=93, num_heads=3):
        super().__init__()
        self.att1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.att2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.pool = nn.AvgPool1d(4, 4)

    def forward(self, x):
        x_att, _ = self.att1(x, x, x)
        x = x + x_att * torch.mean(x_att, dim=-1, keepdim=True)

        x_att, _ = self.att2(x, x, x)
        x = x + x_att * torch.mean(x_att, dim=-1, keepdim=True)

        return self.pool(x)


class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TimeDistributed(nn.Sequential(
            CNNBlock(),
            FeatureAttention()
        ))

    def forward(self, x):
        return self.model(x)


class BottleNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(256, 128, 2, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(128, 32, 2, dilation=2, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 2, dilation=2, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 2, dilation=4, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, 2, dilation=8),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Head(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(256, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),

            nn.Linear(400, 128),
            nn.ReLU(),

            nn.Linear(128, 32),
            nn.ReLU(),

            nn.Linear(32, n_class),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.classifier(x)


class TimeDistributedHead(nn.Module):
    def __init__(self, head_model):
        super().__init__()
        self.model = head_model

    def forward(self, x):
        b, f, t = x.size()
        x = torch.movedim(x, -1, 1)  # shape (b, t, f)
        x = x.contiguous().view(-1, f)
        x = self.model(x)
        return x.view(b, t, -1)


class EEGClassifierModel(nn.Module):
  def __init__(self, n_class=5):
    super().__init__()

    self.backbone = BackBone()
    self.bottleneck = BottleNeck()
    self.timedisthead = TimeDistributedHead(Head(n_class))

  def forward(self, x):
    input_size = x.size()
    # print(input_size)
    x = self.backbone(x)
    # print('BackBone size:            ', x.size())

    x = torch.movedim(x, 1, -2)
    # print('movedim size:             ', x.size())

    x = torch.flatten(x, start_dim=-2)
    # print('flatten size:             ', x.size())

    x = self.bottleneck(x)
    # print('BottleNeck size:          ', x.size())

    x_bottleneck_size = x.size()
    x = x.contiguous().view(x_bottleneck_size[0], x_bottleneck_size[1], input_size[1], -1)
    x = torch.mean(x, dim=-1)
    # print('View size:                ', x.size())

    #x = torch.flatten(x, start_dim=1, end_dim=2)
    #print('flatten size:             ', x.size())

    x = self.timedisthead(x)
    # print('TimeDistributedHead size: ', x.size())
    return x
  

if __name__=="__main__":
    x = torch.zeros(size=(32, 10, 1, 3000))
    # print('Input size: ', x.size(), '\n')

    eeg_model = EEGClassifierModel()
    y = eeg_model(x)

    # print('\nOutput size: ', y.size())