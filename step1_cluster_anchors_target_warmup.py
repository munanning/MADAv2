import os

import faiss
import numpy as np
import torch
import torch.nn.parallel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    seed = 31
    # fix random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # load and transition to 2975*(19*256)
    CAU = torch.load('features/target_full_dataset_objective_vectors_warmup.pkl')
    x = np.reshape(CAU, (CAU.shape[0], CAU.shape[1] * CAU.shape[2])).astype('float32')

    # origin cluster
    ncentroids = 10
    niter = 20
    d = x.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=True)
    kmeans.train(x)
    # get the result
    cluster_centroids = kmeans.centroids
    D, I = kmeans.index.search(x, 1)

    print(len(cluster_centroids))
    print(D.shape, I.shape)

    torch.save(cluster_centroids, 'anchors/cluster_centroids_sub_%d_target_warmup.pkl' % ncentroids)
    torch.save(I, 'anchors/cluster_index_sub_%d_target_warmup.pkl' % ncentroids)


if __name__ == '__main__':
    main()
