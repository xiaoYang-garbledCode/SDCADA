sweep_train_hparams = {}

sweep_alg_hparams = {
        'CoTMix': {
            'learning_rate':            {'values': [5e-3, 3e-3, 1e-3, 7e-3]},
            'temporal_shift':           {'values': [5, 10, 15, 20, 30, 45, 50, 64]},
            'src_cls_weight':           {'distribution': 'uniform', 'min': 1e-1, 'max': 1},
            'mix_ratio':                {'distribution': 'uniform', 'min': 0.5, 'max': 0.99},
            'src_supCon_weight':        {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
            'trg_cont_weight':          {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
            'trg_entropy_weight':       {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
            'teach_trg_entropy_weight': {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
            'loss_disc_dominant_weight': {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
        },

}

