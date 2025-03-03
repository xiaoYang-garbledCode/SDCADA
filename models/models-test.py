from torch import nn
import sys
sys.path.append('D://Users//ygj//Experiment-CoTMix//CoTMix-main_Teach+ATT2//models')
from attention import Seq_Transformer
def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

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

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


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

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

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

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

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
        self.transformer = Seq_Transformer(patch_size=self.patch_size, dim=configs.att_hid_dim, depth=self.depth, heads= self.heads , mlp_dim=self.mlp_dim)
        self.DC = nn.Linear(configs.att_hid_dim, 1)
    def forward(self, input):
        """Forward the discriminator."""
        # src_shape = [batch_size, seq_len, input_dim]
        input = input.view(input.size(0),-1, self.patch_size )
        features = self.transformer(input)
        domain_output = self.DC(features)
        return domain_output

class Discriminator_AR(nn.Module):
    """Discriminator model for source domain."""
    def __init__(self, configs):
        """Init discriminator."""
        self.input_dim = configs.final_out_channels
        super(Discriminator_AR, self).__init__()

        self.AR_disc = nn.GRU(input_size=self.input_dim, hidden_size=configs.disc_AR_hid,num_layers = configs.disc_n_layers,bidirectional=configs.disc_AR_bid, batch_first=True)
        self.DC = nn.Linear(configs.disc_AR_hid+configs.disc_AR_hid*configs.disc_AR_bid, 1)
    def forward(self, input):
        """Forward the discriminator."""
        # src_shape = [batch_size, seq_len, input_dim]
        input = input.view(input.size(0),-1, self.input_dim )
        encoder_outputs, (encoder_hidden) = self.AR_disc(input)
        features = encoder_outputs[:, -1, :]
        domain_output = self.DC(features)
        return domain_output
    def get_parameters(self):
        parameter_list = [{"params":self.AR_disc.parameters(), "lr_mult":0.01, 'decay_mult':1}, {"params":self.DC.parameters(), "lr_mult":0.01, 'decay_mult':1},]
        return parameter_list