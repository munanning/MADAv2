import torch
import numpy as np
import torch.nn.functional as F
import heapq


class Class_Features:
    def __init__(self, centroids, numbers=19):
        self.centroids = centroids
        self.class_numbers = numbers

    def calculate_mean_vector(self, feat_cls, outputs, labels_val, model):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = model.process_label(outputs_argmax.float())

        labels_expanded = model.process_label(labels_val)
        outputs_pred = labels_expanded * outputs_argmax
        print(labels_val.shape, labels_expanded.shape, outputs_pred.shape)
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item() == 0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def calculate_min_mse(self, single_image_objective_vectors):
        loss = []
        for centroid in self.centroids:
            new_loss = np.mean((single_image_objective_vectors - centroid) ** 2)
            loss.append(new_loss)
        min_loss = min(loss)
        min_index = loss.index(min_loss)
        print(min_loss)
        print(min_index)
        return min_loss, min_index


if __name__ == '__main__':
    sub_centroid = torch.load('anchors/cluster_centroids_sub_10.pkl').reshape(10, 19, 256)
    # sub_centroid = torch.load('/remote-home/nmn/MADA_PAMI_laststand/MADA_Dual_dis_far_warmup/anchors/cluster_centroids_sub_10.pkl').reshape(10, 19, 256)
    sub_centroid_target = torch.load('anchors/cluster_centroids_sub_10_target_warmup.pkl').reshape(10, 19, 256)
    target_train_vectors = torch.load('features/target_full_dataset_objective_vectors_warmup.pkl')
    class_features = Class_Features(sub_centroid, numbers=19)
    class_features_target = Class_Features(sub_centroid_target, numbers=19)

    dis_list, idx_list = [], []

    for i in range(len(target_train_vectors)):
        dis, idx = class_features.calculate_min_mse(target_train_vectors[i])
        dis_target, _ = class_features_target.calculate_min_mse(target_train_vectors[i])
        dis_final = dis + dis_target
        dis_list.append(dis_final)
        idx_list.append(idx)

    print(dis_list)
    print(idx_list)
    per = 0.05
    lenth = len(idx_list)
    selected_lenth = round(per * lenth)
    selected_index_list = list(map(dis_list.index, heapq.nlargest(selected_lenth, dis_list)))
    selected_index_list.sort()
    with open('selection_list/stage1_cac_index_0.05_dual_dis_far.txt', 'w') as f:
        for i in range(len(selected_index_list)):
            f.write(str(selected_index_list[i]) + '\n')
    print(selected_index_list)

    with open('data/target_names.txt', 'r') as f:
        target_name_list = f.readlines()

    with open('selection_list/stage1_cac_list_0.05_dual_dis_far.txt', 'w') as f:
        for num in selected_index_list:
            temp = target_name_list[num]
            f.write(temp)
