import datetime
import logging
import os

import torch
import numpy as np
import random


def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def dynamic_copy_paste(images_sup, labels_sup, paste_imgs, paste_labels, query_cat):
    # images_sup, paste_imgs = torch.chunk(images_sup, 2, dim=1)
    # labels_sup, paste_labels = torch.chunk(labels_sup, 2, dim=1)
    labels_sup, paste_labels = labels_sup.squeeze(1), paste_labels.squeeze(1)

    compose_imgs = []
    compose_labels = []
    for idx in range(images_sup.shape[0]):
        paste_label = paste_labels[idx]
        image_sup = images_sup[idx]
        label_sup = labels_sup[idx]
        if torch.sum(paste_label) == 0:
            compose_imgs.append(image_sup.unsqueeze(0))
            compose_labels.append(label_sup.unsqueeze(0))
        else:
            paste_img = paste_imgs[idx]
            alpha = torch.zeros_like(paste_label).int()
            for cat in query_cat:
                alpha = alpha.__or__((paste_label == cat).int())
            alpha = (alpha > 0).int()
            compose_img = (1 - alpha) * image_sup + alpha * paste_img
            compose_label = (1 - alpha) * label_sup + alpha * paste_label
            compose_imgs.append(compose_img.unsqueeze(0))
            compose_labels.append(compose_label.unsqueeze(0))
    compose_imgs = torch.cat(compose_imgs, dim=0)
    compose_labels = torch.cat(compose_labels, dim=0)
    return compose_imgs, compose_labels


def sample_from_bank(cutmix_bank, conf, smooth=False):
    # cutmix_bank [num_classes, len(dataset)]
    conf = (1 - conf).cpu().numpy()
    if smooth:
        conf = conf ** (1 / 3)
    conf = np.exp(conf) / np.sum(np.exp(conf))
    classes = [i for i in range(cutmix_bank.shape[0])]
    class_id = np.random.choice(classes, p=conf)
    sample_bank = torch.nonzero(cutmix_bank[class_id])
    if len(sample_bank) > 0:
        sample_id = random.choice(sample_bank)
    else:
        sample_id = random.randint(0, cutmix_bank.shape[1] - 1)
    return sample_id, class_id
