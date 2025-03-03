def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
                'num_epochs': 40,
                'batch_size': 32,
                'weight_decay': 1e-4,
        }

        self.alg_hparams = {
            'CoTMix': {
                # ori-part1_2
                'learning_rate': 0.005,
                'mix_ratio': 0.66,
                'temporal_shift': 20,
                'src_cls_weight': 0.32,
                'src_supCon_weight': 0.96,
                'trg_cont_weight': 0.29,
                'trg_entropy_weight': 0.19,
                'teach_trg_entropy_weight': 0.30,
                'loss_disc_dominant_weight': 0.57
            }
        }


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
                'num_epochs': 40,
                'batch_size': 32,
                'weight_decay': 1e-4,
        }

        self.alg_hparams = {
            'CoTMix': {
                # 'learning_rate': 0.001,
                # 'mix_ratio': 0.79,
                # 'temporal_shift': 300,
                # 'src_cls_weight': 0.96,
                # 'src_supCon_weight': 0.1,
                # 'trg_cont_weight': 0.1,
                # 'trg_entropy_weight': 0.05,
                # 'teach_trg_entropy_weight': 0.05,
                # 'loss_disc_dominant_weight': 0.05

                # ori-part_1_2 best
                # 'learning_rate': 0.003,
                # 'mix_ratio': 0.96,
                # 'temporal_shift': 300,
                # 'src_cls_weight': 0.81,
                # 'src_supCon_weight': 0.72,
                # 'trg_cont_weight': 0.34,
                # 'trg_entropy_weight': 0.30,
                # 'teach_trg_entropy_weight': 0.06,
                # 'loss_disc_dominant_weight': 0.25

                'learning_rate': 0.003,
                'mix_ratio': 0.6,
                'temporal_shift': 1200,
                'src_cls_weight': 0.8,
                'src_supCon_weight': 0.2, #best 0.2
                'trg_cont_weight': 0.19, # 2024.5.06 ------0.19
                'trg_entropy_weight': 0.01,
                'teach_trg_entropy_weight': 0.55,
                'loss_disc_dominant_weight': 0.33
            }
        }



class WISDM():
    def __init__(self):
        super(WISDM, self).__init__()
        self.train_params = {
                'num_epochs': 40,
                'batch_size': 32,
                'weight_decay': 1e-4,
        }

        self.alg_hparams = {
            # 搜索过了
            'CoTMix': {
                'learning_rate': 0.001,
                'mix_ratio': 0.55,
                'temporal_shift': 50,
                'src_cls_weight': 0.31,
                'src_supCon_weight': 0.13,
                'trg_cont_weight': 0.12,
                'trg_entropy_weight': 0.08,
                'teach_trg_entropy_weight': 0.74,
                'loss_disc_dominant_weight': 0.48
            }
        }


class HHAR():
    def __init__(self):
        super(HHAR, self).__init__()
        self.train_params = {
                'num_epochs': 40,
                'batch_size': 32,
                'weight_decay': 1e-4,
        }

        self.alg_hparams = {
            'CoTMix': {
                # min dev_disk
                # 'learning_rate': 0.001,
                # 'mix_ratio': 0.94,
                # 'temporal_shift': 30,
                # 'src_cls_weight': 0.97,
                # 'src_supCon_weight': 0.75,
                # 'trg_cont_weight': 0.82,
                # 'trg_entropy_weight': 0.37,
                # 'teach_trg_entropy_weight': 0.99,
                # 'loss_disc_dominant_weight': 0.77

                # conditional: min dev_disk    pick: max mf1
                # 'learning_rate': 0.003,
                # 'mix_ratio': 0.51,
                # 'temporal_shift': 10,
                # 'src_cls_weight': 0.78,
                # 'src_supCon_weight': 0.85,
                # 'trg_cont_weight': 0.44,
                # 'trg_entropy_weight': 0.51,
                # 'teach_trg_entropy_weight': 0.35,
                # 'loss_disc_dominant_weight': 0.77

                # ori
                'learning_rate': 0.001,
                'mix_ratio': 0.78,
                'temporal_shift': 10,
                'src_cls_weight': 0.32,
                'src_supCon_weight': 0.15, #ori 0.85   epoch=20
                'trg_cont_weight': 0.09, # ori 0.09
                'trg_entropy_weight': 0.40,
                'teach_trg_entropy_weight': 0.28,
                'loss_disc_dominant_weight': 0.90
            }
        }

class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.train_params = {
                'num_epochs': 40,
                'batch_size': 32,
                'weight_decay': 1e-4,
        }

        self.alg_hparams = {
            'CoTMix': {
                'learning_rate': 0.001,
                'mix_ratio': 0.79,
                'temporal_shift': 300,
                'src_cls_weight': 0.96,
                'src_supCon_weight': 0.5,   # [0.05,0.1,0.2,0.3]  ori:0.1
                'trg_cont_weight': 0.1,
                'trg_entropy_weight': 0.25,
                'teach_trg_entropy_weight': 0.3,
                'loss_disc_dominant_weight': 0.3
            }
        }