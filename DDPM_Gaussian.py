#  Author: Paul-Jason Mello
#  Date: November 30th, 2022

#  General Libraries
import numpy as np
from matplotlib import pyplot as plt

#  Specific Torch
import torch
import torch.utils.data
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torch.distributions.multivariate_normal import MultivariateNormal

#  Misc. Libraries
import datetime
from nvitop import Device  # GPU Monitoring

#  Model Libraries
from NeuralNetwork import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"

devices = Device.all()
for deviceID in devices:
    processes = deviceID.processes()
    sorted_pids = sorted(processes)

# Ensuring CUDA capable
print("Cuda: " + str(torch.cuda.is_available()))
print("Device Count: " + str(torch.cuda.device_count()))
print("Current Device: " + str(torch.cuda.current_device()))
print("Device Info: " + torch.cuda.get_device_name(0))
print("Device: " + str(device))

# Set Seed for Reproducibility
"""
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
"""

"""
Algorithm 1 Training                                                            
1:  repeat  
2:      x_0 ∼ q(x_0)      
3:      t ∼ Uniform({1, . . . , T })    
4:       ∼ N (0, I)      
5:      Take gradient descent step on                                          
            ∇θ ||  − _θ * (√ (̄α_t) * x_0 + √(1−α_t) * , t) || ^ 2        
6: until converged                                                             


Algorithm 2 Sampling                                                            
1: xT ∼ N (0, I)                                                                
2: for t = T, . . . , 1 do                                                      
3:      z ∼ N (0, I) if t > 1, else z = 0                                       
4:      x_t−1 = 1/(√(α_t)) * (x_t − (1−α_t)/√(1−α_t) * _θ(x_t, t)) + σtz      
5: end for
6: return x_0
"""


def main():
    epochs = 50
    lr = 0.0002

    batchSize = 250
    sampleSize = 5_000

    dimensionality = 2
    sampleCount = 5

    steps = 1_000
    t_steps = 1_000

    DDPM = Gaussian_DDPM(steps, t_steps, epochs, batchSize, lr, sampleCount, sampleSize, dimensionality)
    DDPM.run()


class Gaussian_DDPM:
    def __init__(self, steps, t_steps, epochs, batchSize, lr, sampleCount, sampleSize, dimensionality):
        super().__init__()

        self.lr = lr
        self.steps = steps
        self.epochs = epochs
        self.t_steps = t_steps
        self.batchSize = batchSize
        self.sampleCount = sampleCount
        self.sampleSize = sampleSize

        self.loss = float
        self.dimensionality = dimensionality

        self.epochCounter = 0
        self.imageSeries = int(self.t_steps / (self.sampleCount))

        self.Beta = torch.linspace(0.0001, 0.02, self.steps)  # Linearized Variance Schedule
        self.Sqrd_Sigma = self.Beta  # Sigma^2
        self.Alpha = 1.0 - self.Beta  # Alpha

        self.Alpha_Cumprod = torch.cumprod(self.Alpha, dim=0)  # Product Value
        self.Sqrd_Alpha_Cumprod = torch.sqrt(self.Alpha_Cumprod)
        self.Alpha_Cumprod_Previous = F.pad(self.Alpha_Cumprod[:-1], (1, 0), value=1.0)
        self.Sqrd_1_Minus_Alpha_Cumprod = torch.sqrt(1.0 - self.Alpha_Cumprod)
        self.Log_one_minus_Alpha_Cumprod = torch.log(1.0 - self.Alpha_Cumprod)

        # q(x_{t - 1} | x_t, x_0)
        self.Posterior_Variance = self.Beta * (1.0 - self.Alpha_Cumprod_Previous) / (1.0 - self.Alpha_Cumprod)
        self.Posterior_Log_Clamp = torch.log(self.Posterior_Variance.clamp(min=1e-20))
        self.Posterior1 = (self.Beta * torch.sqrt(self.Alpha_Cumprod_Previous) / (1.0 - self.Alpha_Cumprod))
        self.Posterior2 = ((1.0 - self.Alpha_Cumprod_Previous) * torch.sqrt(self.Alpha) / (1.0 - self.Alpha_Cumprod))

    def getModel(self):
        model = NeuralNetwork(self.dimensionality, self.batchSize*2, self.dimensionality, self.t_steps).to(device)
        model = nn.DataParallel(model)  # Parallelize Data when Multi-GPU Applicable (Untested)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, eps=0.00000001)
        return model

    def getMultiGauss(self, mean, covariance):
        sampler = MultivariateNormal(mean, covariance)
        return sampler.sample((self.sampleSize,))

        #X, Y = np.random.multivariate_normal(mean, covariance, self.sampleSize).T
        #plt.scatter(X, Y, c='r')
        #return X, Y

    def getDataset(self):
        #normalize = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda t: (t * 2) - 1)]) #[-1,1]

        mean = np.zeros((1, self.dimensionality))
        covariance = np.diag(np.ones(self.dimensionality))

        print("\nMean Matrix:\n" + str(mean))
        print("\nCov. Matrix:\n" + str(covariance) + "\n")

        meanTensor = torch.from_numpy(np.asarray(mean).astype(float))
        covTensor = torch.from_numpy(np.asarray(covariance).astype(float))
        #return normalize(self.getMultiGauss(meanTensor, covTensor))
        return (self.getMultiGauss(meanTensor, covTensor))

    def getExtract(self, tensor: torch.Tensor, t: torch.Tensor, X):
        out = tensor.gather(-1, t.cpu()).float()
        return out.reshape(t.shape[0], *((1,) * (len(X) - 1))).to(t.device)

    def QMeanVar(self, X, t):
        mean = self.getExtract(self.Sqrd_Alpha_Cumprod, t, X.shape) * X
        variance = self.getExtract(self.Sqrd_1_Minus_Alpha_Cumprod, t, X.shape)
        logVar = self.getExtract(self.Log_one_minus_Alpha_Cumprod, t, X.shape)
        return mean, variance, logVar

    def QPosteriorMeanVar(self, X0, XT, t):  # q(x_{t-1} | x_t, x_0)
        pmean = (self.getExtract(self.Posterior1, t, XT.shape) * X0 + self.getExtract(self.Posterior2, t, XT.shape) * XT)
        pvar = self.getExtract(self.Posterior_Variance, t, XT.shape)
        plog = self.getExtract(self.Posterior_Log_Clamp, t, XT.shape)
        return pmean, pvar, plog

    def predictX0fromXT(self, XT, t, epsilon):
        return (self.getExtract(np.sqrt(1.0 / self.Alpha_Cumprod), t, XT.shape) * XT - self.getExtract(np.sqrt(1.0 / self.Alpha_Cumprod - 1), t, XT.shape) * epsilon)

    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def PMeanVar(self, model, XT, t):
        X0 = self.predictX0fromXT(XT.float(), t, model(XT.float(), t))
        modelMean, pvar, plog = self.QPosteriorMeanVar(X0, XT, t)
        return modelMean, pvar, plog

    def QSample(self, data, idx, t):  # Sample from Q(Xt | X0)
        epsilon = torch.randn_like(data)  #  ∼ N (0, I)

        # Mean, Variance LogVar from Q(Xt | X0)
        mean, variance, logVariance = self.QMeanVar(data, t)

        XT = mean * data + variance.sqrt() * epsilon
        # Sample Images
        return XT.float(), epsilon

    # Sampling
    @torch.no_grad()
    def PSample(self, model, XT, t, tID):  # Getting Sample values at Time T for Processes Q & P
        mean, _, logVar = self.PMeanVar(model, XT, t)
        noise = torch.randn_like(XT)
        nonzeroMask = ((t != 0).float().view(-1, *([1] * (len(XT.shape) - 1))))
        sample = mean + nonzeroMask * torch.exp(0.5 * logVar) * noise
        return sample

    # Looped Sampling for Consistent Reverse Process
    @torch.no_grad()
    def Sample(self, model, img):  # Sampling
        for count in range(self.t_steps):  # SAMPLING 2: for t = T, . . . , 1 do
            t = self.t_steps - count - 1
            # SAMPLING 3 & 4 See Backward PASS (Data = XT or X)
            img = self.PSample(model, img, torch.full((self.batchSize,), t, device=device, dtype=torch.long), t)
        return img  # SAMPLING 6: return generated X_0 Data at End

    @torch.no_grad()
    def plot2DSampleQuality(self, model, XT, dataIndex):
        plt.figure(figsize=(35,8))
        plt.title("Epoch: " + str(self.epochCounter))
        for count in range(0, self.steps)[::-1]:
            XT = self.PSample(model, XT, torch.full((self.batchSize,), count, device=device, dtype=torch.long), count)
            if count == 0 or count % self.imageSeries == 0:
                plt.subplot(1, self.sampleCount, int(self.sampleCount - (count / self.imageSeries)))
                plt.xlim(-750, 750)
                plt.ylim(-750, 750)
                plt.title("T " + str(count))
                for i in range(0, self.batchSize):
                    plt.scatter(np.round(float(XT[i][0][1].cpu()), 4), np.round(float(XT[i][0][0].cpu()), 4), c="b", linewidths=1)

        plt.savefig("Images/Sample Plot Series/E " + str(self.epochCounter) + " T " + str(dataIndex) + ".jpg")
        plt.close()

    def gradientDescent(self, model, idx, X0, t, MI):  # TRAINING 1: Data = 2: X_0 ∼ q(x_0)
        XT, epsilon = self.QSample(X0, idx, t)
        predictedNoise = model(XT, t)

        # MSE(Input, Target) | UNet(XT, t)
        # F.nll_loss only 3D supported, gets 4D
        loss = F.mse_loss(XT, predictedNoise) # TRAINING 5: ∇θ ||  − _θ * (√ (̄α_t) * x_0 + √(1−α_t) * , t) || ^ 2
        return loss, XT, epsilon, predictedNoise


    def TrainingAndSampling(self, model, DataSet, MI):
        flag = False
        # MI in here somewhere
        for idx, Data in enumerate(DataSet):  # TRAINING 1: Repeated Loop
            self.optimizer.zero_grad()

            # TRAINING 3: t ∼ Uniform({1, . . . , T })
            t = torch.randint(0, self.t_steps, (Data.shape[0],), device=device).long()
            # TRAINING 2: X_0 ∼ q(x_0)

            self.loss, XT, epsilon, predNoise = self.gradientDescent(model, idx, Data.to(device), t, MI)

            if int(idx % (len(DataSet)/5)) == 0:
                print(f"T: {idx:05d}/{len(DataSet)}:\tLoss: {self.loss.item()}")

            # Conduct intermittent sampling during training process to demonstrate progress

            if (((self.epochCounter % (self.epochs/2) == 0) or (self.epochCounter == self.epochs-1)) and flag == False and (self.dimensionality == 2)):
                #self.imgGenFromSamples(model, img, "Images/Generated Images/E " + str(self.epochCounter) + " T " + str(int(idx)) + ".jpg")
                self.plot2DSampleQuality(model, XT, idx)  # Plot Sample Transformation over time
                flag = True

            self.loss.backward()
            self.optimizer.step()
        print(f"T: {int((self.sampleSize/self.batchSize)+1):05d}/{len(DataSet)}:\tLoss: {self.loss.item()}")

    def run(self):
        # Get Data
        Data = self.getDataset()
        trainData = DataLoader(dataset=Data, batch_size=self.batchSize, shuffle=True, pin_memory=True)

        # Get Models
        model = self.getModel()
        MI = torch.load("DDPMModels/MINeuralEstimator.pt")

        print("Start Time: " + str(datetime.datetime.now()) + "")

        # Train and Sample
        while self.epochCounter != self.epochs:
            print(f"\n   ------------- Epoch {self.epochCounter} ------------- ")
            self.TrainingAndSampling(model, trainData, MI)  # Sampling done intermittently during training
            self.epochCounter += 1

        torch.save(model, "DDPMModels/GaussianDDPM.pt")

        print(f"\n   ------------- Epoch {self.epochCounter} ------------- ")
        print(f"\t  Final Loss: {self.loss.item()}\n")

        # System Information
        print(f'GPU utilization:\t {deviceID.gpu_utilization()}%')
        print(f'Total memory:\t    {deviceID.memory_total_human()}')
        print(f'Used memory:\t     {deviceID.memory_used_human()}')
        print(f'Free memory:\t     {deviceID.memory_free_human()}')

        print("Completion Time: " + str(datetime.datetime.now()) + "")


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

if __name__ == "__main__":
    main()