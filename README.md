# SDCADA:  Self-supervised deep contrastive and auto-regressive domain adaptation for time-series based on channel recalibration [[Paper](https://doi.org/10.1016/j.engappai.2025.110280)] [[Cite](#citation)]

### This work has been accepted for publication in the Engineering Applications of Artificial Intelligence(EAAI).

### The code and the experimental datasets are adopted from [AdaTime framework](https://github.com/emadeldeen24/AdaTime)

## Abstract
<p align="center">
<img src="misc/SDCADA.jpg" width="800" class="center">
</p>


Time-series based unsupervised domain adaptation (UDA) techniques have been widely adopted to the application of intelligent systems, such as sleep staging, fault diagnosis, and human activity recognition. However,  recently methods have overlooked the importance of temporal feature representations and the distribution  discrepancies across domains, which deteriorated UDA performance. To address these challenges, we proposed a  novel Self-supervised Deep Contrastive and Auto-regressive Domain Adaptation (SDCADA) model for crossdomain time-series classification. Specifically, the cross-domain mixup preprocessing strategy is applied to  reduce sample-level distribution discrepancy, then we proposed to introduce the channel recalibration module  for adaptively selecting discriminative representations. Afterwards, the auto-regressive discriminator and teacher  model are proposed to reduce the distribution discrepancies of feature representations. Finally, a total of six  losses, including contrastive and adversarial learning, are weighted and jointly optimized to train the SDCADA  model. The proposed SDCADA model has been systematically experimented on four cross-domain time-series  benchmarked datasets, and its classification performance surpasses several recently proposed state-of-the-art  models. Moreover, it effectively captures discriminative and comprehensive cross-domain time-series feature  representations with parameter insensitivity.

## Results
<p align="center">
<img src="misc/RESULT1.jpg" width="900" class="center">
</p>


## Citation
If you found this work useful for you, please consider citing it.
```
@ARTICLE{xiaoYang-garbledCode2025SDCADA,
  author={Guangju Yang,Tian-jian Luo,Xiaochen Zhang},
  journal={Engineering Applications of Artificial Intelligence}, 
  title={Self-supervised deep contrastive and auto-regressive domain adaptation for time-series based on channel recalibration}, 
  year={2025},
  volume={},
  number={},
  pages={1-19},
  doi={10.1016/j.engappai.2022.105375}
}
```

## Contact
For any issues/questions regarding the paper or reproducing the results, please contact me.   
Guangju Yang   
School of Computer Science and Engineering (SCSE),   
Fujian Normal University Fuzhou China.   
Email: 1747749798@qq.com 
