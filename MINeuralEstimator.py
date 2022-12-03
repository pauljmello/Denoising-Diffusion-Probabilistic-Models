import torch
import numpy as np
from matplotlib import pyplot as plt

from pytorch_lightning import Trainer
from mine import MutualInformationEstimator
from MultivariateNormalDataset import MultivariateNormalDataset


device = "cuda" if torch.cuda.is_available() else "cpu"


dim = 5
N = 1_000
lr = 1e-5
epochs = 25
batch_size = 32

rhos = np.linspace(-1, 1, 20)
loss_type = ['mine_biased']

results_dict = dict()

for loss in loss_type:
    results = []
    for rho in rhos:
        train_loader = torch.utils.data.DataLoader(MultivariateNormalDataset(N, dim, rho), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(MultivariateNormalDataset(N, dim, rho), batch_size=batch_size, shuffle=True)

        true_mi = train_loader.dataset.true_mi

        kwargs = {
            'lr': lr,
            'batch_size': batch_size,
            'train_loader': train_loader,
            'test_loader': test_loader,
            'alpha': 1.0
        }

        model = MutualInformationEstimator(dim, dim, loss=loss, **kwargs).to(device)

        trainer = Trainer(max_epochs=epochs,  gpus=1)
        trainer.fit(model)
        trainer.test()

        print("True_mi {}".format(true_mi))
        print("MINE {}".format(model.avg_test_mi))
        results.append((rho, model.avg_test_mi, true_mi))

    results = np.array(results)
    results_dict[loss] = results

torch.save(model, "DDPMModels/MINEGaussianModel.pt")

fig, axs = plt.subplots(1, len(loss_type), figsize=(15, 5))
plots = []
for ix, loss in enumerate(loss_type):
    results = results_dict[loss]
    plots += axs.plot(results[:, 0], results[:, 1], label='MINE')
    plots += axs.plot(results[:, 0], results[:, 2], linestyle='--', label='True MI')
    axs.set_xlabel('correlation')
    axs.set_ylabel('mi')
    # axs.title.set_text(f"{loss} for {dim} dimensional inputs")

fig.legend(plots[0:2], labels=['MINE', 'True MI'], loc='upper right')
fig.savefig('figures/mi_estimation.png')