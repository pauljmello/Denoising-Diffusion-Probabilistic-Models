import torch
import numpy as np

from torch.distributions import MultivariateNormal

def to_onehot(labels, num_points):
    num_classes = len(torch.unique(labels))
    one_hot = torch.zeros((num_points, num_classes))
    one_hot[torch.arange(num_points), labels.long()] = 1.0

    return one_hot

class MultivariateNormalDataset(torch.utils.data.Dataset):
    def __init__(self, N, dim, rho):
        self.N = N
        self.rho = rho
        self.dim = dim

        self.dist = self.build_dist
        self.x = self.dist.sample((N, ))
        self.dim = dim

    def __getitem__(self, ix):
        a, b = self.x[ix, 0:self.dim], self.x[ix, self.dim:2 * self.dim]
        return a, b

    def __len__(self):
        return self.N

    @property
    def build_dist(self):
        mu = torch.zeros(2 * self.dim)
        dist = MultivariateNormal(mu, self.cov_matrix)
        return dist

    @property
    def cov_matrix(self):
        cov = torch.zeros((2 * self.dim, 2 * self.dim))
        cov[torch.arange(self.dim), torch.arange(
            start=self.dim, end=2 * self.dim)] = self.rho
        cov[torch.arange(start=self.dim, end=2 * self.dim),
            torch.arange(self.dim)] = self.rho
        cov[torch.arange(2 * self.dim), torch.arange(2 * self.dim)] = 1.0

        return cov

    @property
    def true_mi(self):
        return -0.5 * np.log(np.linalg.det(self.cov_matrix.data.numpy()))

class Gaussians(torch.utils.data.Dataset):

    def __init__(self, n_points, std=1.0):
        self.n = n_points
        distance = 3 * 1.41
        centers = []
        for i in range(5):
            for j in range(5):
                center = [i * distance, j * distance]
                centers.append(center)

        x = []
        labels = []

        for ix, center in enumerate(centers):
            rand_n = np.random.multivariate_normal(
                center, np.eye(2) * std**2, size=n_points)
            label = np.ones(n_points) * ix
            x.extend(rand_n)
            labels.extend(label)

        x = np.asarray(x)
        self.labels = np.asarray(labels)

        # normalize
        self.x_np = (x - np.mean(x)) / np.std(x)

        self.x = torch.from_numpy(self.x_np).float()
        self.labels_onehot = to_onehot(
            torch.from_numpy(self.labels), len(self.labels)).float()
