# 作者: Peach Yang
# 2023年12月21日14时52分47秒
import torch
from torch import nn
import sys

sys.path.append('D://Users//ygj//Experiment-CoTMix//CoTMix-main_Teach+ATT2//models')
from torch import nn
from attention import Seq_Transformer


def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        filter_sizes = [5, 9, 11]
        self.conv1 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[0],
                               stride=configs.stride, bias=False, padding=(filter_sizes[0] // 2))
        self.conv2 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[1],
                               stride=configs.stride, bias=False, padding=(filter_sizes[1] // 2))
        self.conv3 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[2],
                               stride=configs.stride, bias=False, padding=(filter_sizes[2] // 2))

        self.bn = nn.BatchNorm1d(configs.mid_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.do = nn.Dropout(configs.dropout)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.inplanes = 128
        self.crm = self._make_layer(SEBasicBlock, 128, 3)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=configs.trans_dim, nhead=configs.num_heads,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.aap = nn.AdaptiveAvgPool1d(configs.features_len)




    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_in):
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        x3 = self.conv3(x_in)

        x_concat = torch.mean(torch.stack([x1, x2, x3], 2), 2)
        x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))

        x = self.conv_block2(x_concat)
        x = self.conv_block3(x)

        # Channel Recalibration Module
        x = self.crm(x)

        # Bi-directional Transformer
        # x1 = self.transformer_encoder(x)
        # x2 = self.transformer_encoder(torch.flip(x, [2]))
        # x = x1 + x2

        x = self.aap(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()
        self.logits = nn.Linear(configs.features_len * configs.final_out_channels, configs.num_classes)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions


class teach_Model(nn.Module):
    def __init__(self, configs):
        super(teach_Model, self).__init__()

        filter_sizes = [5, 9, 11]
        self.conv1 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[0],
                               stride=configs.stride, bias=False, padding=(filter_sizes[0] // 2))
        self.conv2 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[1],
                               stride=configs.stride, bias=False, padding=(filter_sizes[1] // 2))
        self.conv3 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[2],
                               stride=configs.stride, bias=False, padding=(filter_sizes[2] // 2))

        self.bn = nn.BatchNorm1d(configs.mid_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.do = nn.Dropout(configs.dropout)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.inplanes = 128
        self.crm = self._make_layer(SEBasicBlock, 128, 3)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=configs.trans_dim, nhead=configs.num_heads,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.aap = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_in):
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        x3 = self.conv3(x_in)

        x_concat = torch.mean(torch.stack([x1, x2, x3], 2), 2)
        x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))

        x = self.conv_block2(x_concat)
        x = self.conv_block3(x)

        # Channel Recalibration Module
        x = self.crm(x)

        # Bi-directional Transformer
        x1 = self.transformer_encoder(x)
        x2 = self.transformer_encoder(torch.flip(x, [2]))
        x = x1 + x2

        x = self.aap(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


class Discriminator_ATT(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        self.patch_size = configs.patch_size
        self.hid_dim = configs.att_hid_dim
        self.depth = configs.depth
        self.heads = configs.heads
        self.mlp_dim = configs.mlp_dim
        super(Discriminator_ATT, self).__init__()
        self.transformer = Seq_Transformer(patch_size=self.patch_size, dim=configs.att_hid_dim, depth=self.depth,
                                           heads=self.heads, mlp_dim=self.mlp_dim)
        self.DC = nn.Linear(configs.att_hid_dim, 1)

    def forward(self, input):
        """Forward the discriminator."""
        # src_shape = [batch_size, seq_len, input_dim]
        input = input.view(input.size(0), -1, self.patch_size)
        features = self.transformer(input)
        domain_output = self.DC(features)
        return domain_output


class Discriminator_AR(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        self.input_dim = configs.final_out_channels
        super(Discriminator_AR, self).__init__()

        self.AR_disc = nn.GRU(input_size=self.input_dim, hidden_size=configs.disc_AR_hid,
                              num_layers=configs.disc_n_layers, bidirectional=configs.disc_AR_bid, batch_first=True)
        self.DC = nn.Linear(configs.disc_AR_hid + configs.disc_AR_hid * configs.disc_AR_bid, 1)

    def forward(self, input):
        """Forward the discriminator."""
        # src_shape = [batch_size, seq_len, input_dim]
        input = input.view(input.size(0), -1, self.input_dim)
        encoder_outputs, (encoder_hidden) = self.AR_disc(input)
        features = encoder_outputs[:, -1, :]
        domain_output = self.DC(features)
        return domain_output

    def get_parameters(self):
        parameter_list = [{"params": self.AR_disc.parameters(), "lr_mult": 0.01, 'decay_mult': 1},
                          {"params": self.DC.parameters(), "lr_mult": 0.01, 'decay_mult': 1}, ]
        return parameter_list
