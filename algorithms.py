import torch
import torch.nn as nn
import numpy as np
import time

from models.models import classifier
from models.loss import SupConLoss, ConditionalEntropyLoss, NTXentLoss


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class CoTMix(Algorithm):
    def __init__(self, backbone_fe, teach_Model, Discriminator_Model,configs, hparams, device):
        super(CoTMix, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.teach_Model = teach_Model(configs)
        self.feature_discriminator = Discriminator_Model(configs)
        self.network = nn.Sequential(self.feature_extractor, self.feature_discriminator,self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

        self.contrastive_loss = NTXentLoss(device, hparams["batch_size"], 0.2, True)
        self.entropy_loss = ConditionalEntropyLoss()
        self.sup_contrastive_loss = SupConLoss(device)
        self.criterion_disc = nn.BCEWithLogitsLoss()
        self.device = device

    def update(self, src_x, src_y, trg_x):

        # ====== Temporal Mixup =====================
        start_time = time.time()
        mix_ratio = round(self.hparams.mix_ratio, 2)
        temporal_shift = self.hparams.temporal_shift
        h = temporal_shift // 2  # half

        src_dominant = mix_ratio * src_x + (1 - mix_ratio) * \
                       torch.mean(torch.stack([torch.roll(trg_x, -i, 2) for i in range(-h, h)], 2), 2)

        trg_dominant = mix_ratio * trg_x + (1 - mix_ratio) * \
                       torch.mean(torch.stack([torch.roll(src_x, -i, 2) for i in range(-h, h)], 2), 2)

        # ====== Extract features and calc logits =====================
        self.optimizer.zero_grad()

        # Src original features
        src_orig_feat = self.feature_extractor(src_x)
        src_orig_logits = self.classifier(src_orig_feat)

        # Target original features
        trg_orig_feat = self.feature_extractor(trg_x)
        trg_orig_logits = self.classifier(trg_orig_feat)

        # -----------  The two main losses: L_CE on source and L_ent on target
        # Cross-Entropy loss
        src_cls_loss = self.cross_entropy(src_orig_logits, src_y)
        loss = src_cls_loss * round(self.hparams.src_cls_weight, 2)

        # Target Entropy loss
        trg_entropy_loss = self.entropy_loss(trg_orig_logits)
        loss += trg_entropy_loss * round(self.hparams.trg_entropy_weight, 2)

        # ====== teacher model =====================
        # Target original features
        teach_trg_orig_feat = self.teach_Model(trg_x)
        teach_trg_orig_logits = self.classifier(teach_trg_orig_feat)
        # --------------- 在目标域中找置信预测与标签
        softmax = nn.Softmax(dim=1)  # 使用 softmax 函数，以将预测转换为概率分布。
        trg_normalized_pred = softmax(teach_trg_orig_logits)
        pred_prob = trg_normalized_pred.max(1, keepdim=True)[0].squeeze()  # 找到每个样本中概率最大的类别的概率值
        target_pseudo_labels = trg_normalized_pred.max(1, keepdim=True)[1].squeeze()
        confidence_level = 0.9  # 设置置信程度
        teach_trg_confident_pred = teach_trg_orig_logits[pred_prob > confidence_level]
        teach_trg_confident_labels = target_pseudo_labels[pred_prob > confidence_level]
        # 如果存在置信程度大于0.9的标签，则计算类损失
        if (teach_trg_confident_pred.shape[0] != 0):
            teach_trg_entroy_loss = self.cross_entropy(teach_trg_confident_pred, teach_trg_confident_labels)
            loss += teach_trg_entroy_loss * round(self.hparams.teach_trg_entropy_weight, 2)

        # ====== discirminator =====================
        # concatenate source and target features
        feat_concat = torch.cat((src_orig_feat, trg_orig_feat), dim=0)
        # predict the domain label by the discirminator network
        pred_concat = self.feature_discriminator(feat_concat.detach())
        # prepare real labels for the training the discriminator
        # 在域对抗学习中，通常使用 label_src 表示源域的标签（通常为1），label_tgt 表示目标域的标签（通常为0），
        label_src = torch.ones(src_orig_feat.size(0)).to(self.device)
        label_tgt = torch.zeros(trg_orig_feat.size(0)).to(self.device)
        label_concat = torch.cat((label_src, label_tgt), 0)
        # criterion_disc 通常用于二元分类问题，它的目标是最小化模型的输出与真实标签之间的二元交叉熵损失。
        # Discriminator loss
        loss_Discriminator = self.criterion_disc(pred_concat.squeeze(), label_concat.float())
        loss += loss_Discriminator * round(self.hparams.loss_disc_dominant_weight, 2)

        # -----------  Auxiliary losses
        # Extract source-dominant mixup features.
        src_dominant_feat = self.feature_extractor(src_dominant)
        src_dominant_logits = self.classifier(src_dominant_feat)

        # supervised contrastive loss on source domain side
        src_concat = torch.cat([src_orig_logits.unsqueeze(1), src_dominant_logits.unsqueeze(1)], dim=1)
        src_supcon_loss = self.sup_contrastive_loss(src_concat, src_y)
        # src_con_loss = self.contrastive_loss(src_orig_logits, src_dominant_logits) # unsupervised_contrasting
        loss += src_supcon_loss * round(self.hparams.src_supCon_weight, 2)

        # Extract target-dominant mixup features.
        trg_dominant_feat = self.feature_extractor(trg_dominant)
        trg_dominant_logits = self.classifier(trg_dominant_feat)

        # Unsupervised contrastive loss on target domain side
        trg_con_loss = self.contrastive_loss(trg_orig_logits, trg_dominant_logits)
        loss += trg_con_loss * round(self.hparams.trg_cont_weight, 2)

        loss.backward()
        self.optimizer.step()

        # 指数移动平均（EMA）更新，将目标模型（target_model）的参数平滑地更新到教师模型（teacher_model）
        alpha = 0.9
        for mean_param, param in zip(self.teach_Model.parameters(), self.feature_extractor.parameters()):
            mean_param.data.mul_(alpha).add_(1 - alpha, param.data)

        end_time = time.time()

        if (teach_trg_confident_pred.shape[0] != 0):
            return {'Total_loss': loss.item(),
                    'src_cls_loss': src_cls_loss.item(),
                    'trg_entropy_loss': trg_entropy_loss.item(),
                    'src_supcon_loss': src_supcon_loss.item(),
                    'trg_con_loss': trg_con_loss.item(),
                    'teach_trg_entroy_loss:': teach_trg_entroy_loss.item(),
                    'loss_disc_dominant_weight': loss_Discriminator.item()
                    }
        else:
            return {'Total_loss': loss.item(),
                    'src_cls_loss': src_cls_loss.item(),
                    'trg_entropy_loss': trg_entropy_loss.item(),
                    'src_supcon_loss': src_supcon_loss.item(),
                    'trg_con_loss': trg_con_loss.item(),
                    'loss_disc_dominant_weight': loss_Discriminator.item()
                    }
