# MADAv2
Project for [MADAv2: Advanced Multi-Anchor Based Active Domain Adaptation Segmentation](https://arxiv.org/abs/2301.07354) (accepted by TPAMI), which is modified from [Multi-Anchor Active Domain Adaptation for Semantic Segmentation](https://arxiv.org/abs/2108.08012) (ICCV Oral 2021).

> **Abstract.**
> Unsupervised domain adaption has been widely adopted in tasks with scarce annotated data.
> Unfortunately, mapping the target-domain distribution to the source-domain unconditionally may distort the essential structural information of the target-domain data, leading to inferior performance.
To address this issue, we firstly propose to introduce active sample selection to assist domain adaptation regarding the semantic segmentation task.
> By innovatively adopting multiple anchors instead of a single centroid, both source and target domains can be better characterized as multimodal distributions, in which way more complementary and informative samples are selected from the target domain.
> With only a little workload to manually annotate these active samples, the distortion of the target-domain distribution can be effectively alleviated, achieving a large performance gain.
> In addition, a powerful semi-supervised domain adaptation strategy is proposed to alleviate the long-tail distribution problem and further improve the segmentation performance.
> Extensive experiments are conducted on public datasets, and the results demonstrate that the proposed approach outperforms state-of-the-art methods by large margins and achieves similar performance to the fully-supervised upperbound, *i.e.*, 71.4\% mIoU on GTA5 and 71.8\% mIoU on SYNTHIA.
> The effectiveness of each component is also verified by thorough ablation studies. 

![](./img/visualization.png)
As shown in the figure, our features are perfectly distributed around the target centers, while traditional features of adversarial training tend to deviate from the real target distribution.

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)
- [Notes](#notes)

## Requirements

The code requires Pytorch >= 0.4.1 and faiss-cpu >= 1.7.2. The code is trained using a NVIDIA RTX3090 with 24GB memory.

## Usage

1. Preparation:
* Download the [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) dataset as the source domain, and the [Cityscapes](https://www.cityscapes-dataset.com/) dataset as the target domain.
* Download the [Weights](https://drive.google.com/drive/folders/1Ln-fTBTivmMGJdRiVOi1774eBK_GMrhZ?usp=sharing) and [Features](https://drive.google.com/drive/folders/17DMUHU97X5JPnEi9Hx8xWv-YYRDKdfie?usp=sharing). Move features to the MADAv2 directory.

2. Set up the config files.
* Set the data paths
* Set the pretrained model paths

3. Quickstart
* To run the code with our weights and anchors:
~~~~
python3 step1_train_active_sup_only.py
python3 step2_train_active_semi_sup.py
~~~~
* During the training, the generated files (log file) will be written in the folder 'runs/..'.

4. Evaluation
* Set the config file for test (configs/test_from_city_to_gta.yml):
* Run test.py to see the results:
~~~~
python3 test.py
~~~~

5. Training-whole process
* Setting the config files.
* Stage 1:
* 1-Save the features for source and target domains with the warmup model:
~~~~
python3 step1_save_feat_source.py
python3 step1_save_feat_target_warmup.py
~~~~
* 2-Cluster the features of source and target domains:
~~~~
python3 step1_cluster_anchors_source.py
python3 step1_cluster_anchors_target_warmup.py
~~~~
* 3-Select the active samples by considering the distance from the both domains:
~~~~
python3 step1_select_active_samples.py
~~~~
* 4-Training with the active samples:
~~~~
python3 step1_train_active_sup_only.py
~~~~

* Stage 2:
* 1-Save the features of target samples with the stage1 model:
~~~~
python3 step2_save_feat_target.py
~~~~
* 2-Cluster the features of target samples:
~~~~
python3 step2_cluster_anchors_target.py
~~~~
* 3-Training with the proposed semi-supervised domain adaptation strategy: 
~~~~
python3 step2_train_active_semi_sup.py
~~~~



## License

[MIT](LICENSE)

The code is heavily borrowed from the CAG_UDA (https://github.com/RogerZhangzz/CAG_UDA) and U2PL (https://github.com/Haochen-Wang409/U2PL).

If you use this code and find it usefule, please cite:
~~~~
@article{ning2023madav2,
  title={MADAv2: Advanced Multi-Anchor Based Active Domain Adaptation Segmentation},
  author={Ning, Munan and Lu, Donghuan and Xie, Yujia and Chen, Dongdong and Wei, Dong and Zheng, Yefeng and Tian, Yonghong and Yan, Shuicheng and Yuan, Li},
  journal={arXiv preprint arXiv:2301.07354},
  year={2023}
}
~~~~

## Notes
We also provide the results of D2ADA version in [Weights_D2ADA](https://drive.google.com/drive/folders/1pnSJZ-WWkYivdRokD9rteyPQ4DVWeGcu?usp=sharing).

As you see, our framework is kind of out of date. If you want to continue in the research of domain adaptation, we recommend you to use the [D2ADA](https://github.com/tsunghan-wu/D2ADA) framework, which is more powerful and easy to use.
