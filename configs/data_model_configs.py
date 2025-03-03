
def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HAR():
    def __init__(self):
        super(HAR, self)
        self.scenarios = [("2", "11"), ("12", "16"), ("9", "18"),("6", "23"), ("7", "13"), ]
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # CNN features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1
        # Att Disc Parameters
        self.att_hid_dim = 64
        self.patch_size = 128
        self.depth = 8
        self.heads = 2
        self.mlp_dim = 64

        self.trans_dim = 18
        self.num_heads = 3
class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        # data parameters
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.scenarios = [("16", "1"), ("9", "14"), ("12", "5"), ("7", "18"), ("0", "11"), ]
        self.sequence_len = 3000

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2
        self.num_classes = 5

        # features
        self.mid_channels = 32
        self.final_out_channels = 128
        self.features_len = 1
        # GRU configs
        self.disc_n_layers = 1
        self.disc_AR_hid = 512
        self.disc_AR_bid = False
        self.disc_hid_dim = 100
        self.disc_out_dim = 1

        self.trans_dim = 65
        self.num_heads = 5

class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.scenarios = [("35", "31"), ("7", "18"), ("20", "30"), ("6", "19"), ("18", "23"), ]
        self.sequence_len = 128

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1
        # Att Disc Parameters
        self.att_hid_dim = 64
        self.patch_size = 128
        self.depth = 8
        self.heads = 2
        self.mlp_dim = 64

        self.trans_dim = 18
        self.num_heads = 3
class HHAR(object):
    def __init__(self):
        super(HHAR, self).__init__()
        self.scenarios = [("0", "6"), ("1", "6"), ("2", "7"), ("3", "8"), ("4", "5")]
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.sequence_len = 128
        
        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1
        # Att Disc Parameters
        self.att_hid_dim = 64
        self.patch_size = 128
        self.depth = 8
        self.heads = 2
        self.mlp_dim = 64

        # Transformer
        self.trans_dim = 18
        self.num_heads = 6

class FD(object):
    def __init__(self):
        super(FD, self).__init__()
        self.scenarios = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "a"), ("b", "c"), ("b", "d"), ("c", "a"),
                          ("c", "b"), ("c", "d"), ("d", "a"), ("d", "b"), ("d", "c")]
        self.class_names = ['internal_fault', 'external_fault', 'health']
        self.sequence_len = 128

        # model configs
        self.input_channels = 1
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 3

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # GRU configs
        self.disc_n_layers = 1
        self.disc_AR_hid = 512
        self.disc_AR_bid = False
        self.disc_hid_dim = 100
        self.disc_out_dim = 1

        # Transformer
        self.trans_dim = 65
        self.num_heads = 5