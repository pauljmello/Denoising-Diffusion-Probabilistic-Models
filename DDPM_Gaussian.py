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
from torch.utils.data import DataLoader
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

SEED = 3407  # https://arxiv.org/abs/2109.08203
np.random.seed(SEED)
torch.manual_seed(SEED)

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
    epochs = 500
    lr = 2e-4
    batchSize = 250
    schedule = "linear"     # "linear", "cosine" https://arxiv.org/abs/2102.09672

    sampleSize = 30_000

    dimensionality = 2
    sampleCount = 5

    steps = 1_000
    t_steps = 1_000

    DDPM = DDPM_Gaussian(steps, t_steps, epochs, batchSize, lr, sampleCount, sampleSize, dimensionality, schedule)
    DDPM.run()


class DDPM_Gaussian:
    def __init__(self, steps, t_steps, epochs, batchSize, lr, sampleCount, sampleSize, dimensionality, schedule):
        super().__init__()

        self.lr = lr
        self.steps = steps
        self.epochs = epochs
        self.t_steps = t_steps
        self.batchSize = batchSize
        self.sampleSize = sampleSize
        self.sampleCount = sampleCount - 1
        self.dimensionality = dimensionality

        self.loss = float
        self.epochCounter = 0
        self.imageSeries = int(self.t_steps / (self.sampleCount))

        self.Beta = self.getSchedule(schedule)  # Schedule
        self.Sqrd_Sigma = self.Beta  # Sigma Squared
        self.Alpha = 1.0 - self.Beta  # Alpha

        self.Alpha_Cumprod = torch.cumprod(self.Alpha, dim=0)  # Product Value of Alpha
        self.Sqrd_Alpha_Cumprod = torch.sqrt(self.Alpha_Cumprod)  # Square Root of Product Value of Alpha
        self.Alpha_Cumprod_Previous = F.pad(self.Alpha_Cumprod[:-1], (1, 0), value=1.0)  # Previous Product Value of Alpha
        self.Sqrd_1_Minus_Alpha_Cumprod = torch.sqrt(1.0 - self.Alpha_Cumprod)  # Square Root of 1 - Product Value of Alpha
        self.Log_one_minus_Alpha_Cumprod = torch.log(1.0 - self.Alpha_Cumprod)  # Log of 1 - Product Value of Alpha

        self.Sqrd_Recipricol_Alpha_Cumprod = torch.sqrt(1 / self.Alpha_Cumprod)  # Square Root of Reciprocal of Product Value of Alpha
        self.Sqrd_Recipricol_Alpha_Cumprod_Minus_1 = torch.sqrt(1 / self.Alpha_Cumprod-1)  # Square Root of Reciprocal of Product Value of Alpha - 1

        # q(x_{t - 1} | x_t, x_0)
        self.Posterior_Variance = self.Beta * (1.0 - self.Alpha_Cumprod_Previous) / (1.0 - self.Alpha_Cumprod)  # Var(x_{t-1} | x_t, x_0)
        self.Posterior_Log_Clamp = np.log(np.maximum(self.Posterior_Variance, 1e-20))  # Log of Var(x_{t-1} | x_t, x_0)
        self.Posterior1 = (self.Beta * torch.sqrt(self.Alpha_Cumprod_Previous) / (1.0 - self.Alpha_Cumprod))  # 1 / (Var(x_{t-1} | x_t, x_0))
        self.Posterior2 = ((1.0 - self.Alpha_Cumprod_Previous) * torch.sqrt(self.Alpha) / (1.0 - self.Alpha_Cumprod))  # (1 - Alpha_{t-1}) / (Var(x_{t-1} | x_t, x_0))

    def getSchedule(self, schedule):
        if schedule == "linear":
            return torch.linspace(1e-4, 2e-2, self.steps)
        elif schedule == "cosine":
            return self.getCosineBeta()

    def getCosineBeta(self):
        x = torch.linspace(0, self.t_steps, self.steps + 1)
        y = torch.cos(((x / self.t_steps) + 0.01) / (1 + 0.01) * torch.pi * 0.5) ** 2
        z = y / y[0]
        return torch.clip((1 - (z[1:] / z[:-1])), 0.0001, 0.9999)

    def getModel(self):
        model = NeuralNetwork(self.dimensionality, self.batchSize, self.dimensionality, self.t_steps, 128).to(device)
        model = nn.DataParallel(model)  # Parallelize Data when Multi-GPU Applicable (Untested)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, eps=1e-8)
        return model

    def getDataset(self):
        mean = np.zeros((1, self.dimensionality))
        covariance = np.diag(np.ones(self.dimensionality))

        print("\nMean Matrix:\n" + str(mean))
        print("\nCov. Matrix:\n" + str(covariance) + "\n")

        meanTensor = torch.from_numpy(np.asarray(mean).astype(float))
        covTensor = torch.from_numpy(np.asarray(covariance).astype(float))
        #return normalize(self.getMultiGauss(meanTensor, covTensor))
        return (self.getMultiGauss(meanTensor, covTensor))

    def getMultiGauss(self, mean, covariance):
        sampler = MultivariateNormal(mean, covariance)
        return sampler.sample((self.sampleSize,)) # Sample from Multivariate Normal Distribution

    def getExtract(self, tensor: torch.Tensor, t: torch.Tensor, X):
        out = tensor.gather(-1, t.cpu()).float()
        return out.reshape(t.shape[0], *((1,) * (len(X) - 1))).to(t.device) # Reshape to match X

    def q_mean_var(self, X0, t):
        X0_shape = X0.shape
        mean = self.getExtract(self.Sqrd_Alpha_Cumprod, t, X0_shape) * X0
        variance = self.getExtract(1.0 - self.Alpha_Cumprod, t, X0_shape)
        logVar = self.getExtract(self.Log_one_minus_Alpha_Cumprod, t, X0_shape)
        return mean, variance, logVar

    def q_posterior_mean_variance(self, X0, XT, t):  # q(x_{t-1} | x_t, x_0)
        XT_shape = XT.shape
        posterior_mean = self.getExtract(self.Posterior1, t, XT_shape) * X0 + self.getExtract(self.Posterior2, t, XT_shape) * XT
        posterior_var = self.getExtract(self.Posterior_Variance, t, XT_shape)
        posterior_log = self.getExtract(self.Posterior_Log_Clamp, t, XT_shape)
        return posterior_mean, posterior_var, posterior_log

    def q_sample(self, data, idx, t):  # Sample from Q(Xt | X0)
        epsilon = torch.randn_like(data)  #  ∼ N (0, I)

        # Mean, Variance LogVar from Q(Xt | X0)
        # mean, variance, logVariance = self.q_mean_var(data, t)
        # XT = mean * data + variance.sqrt() * epsilon
        # Sample Images
        # return XT.float(), epsilon

        return (self.getExtract(self.Sqrd_Alpha_Cumprod, t, data.shape) * data
                + self.getExtract(self.Sqrd_1_Minus_Alpha_Cumprod, t, data.shape) * epsilon).float(), epsilon # Sample Images

    def pred_X0_from_XT(self, XT, t, epsilon):
        XT_shape = XT.shape
        return (self.getExtract(self.Sqrd_Recipricol_Alpha_Cumprod, t, XT_shape) * XT - self.getExtract(self.Sqrd_Recipricol_Alpha_Cumprod_Minus_1, t, XT_shape) * epsilon) # Sample Images

    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, XT, t):
        X0_prediction = self.pred_X0_from_XT(XT.float(), t, model(XT.float(), t)) # p(x_{t-1} | x_t)
        model_mean, posterior_Variance, posterior_log_variance = self.q_posterior_mean_variance(X0_prediction, XT, t) # q(x_{t-1} | x_t, x_0)
        return model_mean, posterior_Variance, posterior_log_variance

    # Sampling
    @torch.no_grad()
    def p_sample(self, model, XT, t, tID):  # Getting Sample values at Time T for Processes Q & P
        model_mean, _, model_logVar = self.p_mean_variance(model, XT, t)
        epsilon = torch.randn_like(XT)
        # No noise at t == 0
        nonzeroMask = ((t != 0).float().view(-1, *([1] * (len(XT.shape) - 1)))) # Mask for t != 0
        sample = model_mean + nonzeroMask * torch.exp(0.5 * model_logVar) * epsilon # Sample from P(Xt | X0)
        return sample

    # Looped Sampling for Consistent Reverse Process
    # Unused Method
    # Sampling
    @torch.no_grad()
    def sample(self, model, img):
        for count in range(self.t_steps):  # SAMPLING 2: for t = T, . . . , 1 do
            t = self.t_steps - count - 1
            # SAMPLING 3 & 4 See Backward PASS (Data = XT or X)
            img = self.p_sample(model, img, torch.full((self.batchSize,), t, device=device, dtype=torch.long), t) # Sample from P(Xt | X0)
        return img  # SAMPLING 6: return generated X_0 Data at End

    @torch.no_grad()
    def plot_2D_gaussian(self, model, XT, dataIndex):
        plt.figure(figsize=(35,8))
        plt.title("Epoch: " + str(self.epochCounter))
        for count in range(0, self.steps)[::-1]:
            XT = self.p_sample(model, XT, torch.full((self.batchSize,), count, device=device, dtype=torch.long), count)
            if count == 0 or count % self.imageSeries == 0:
                self.plot_scatter(XT, count, count)
            elif count == 999:
                self.plot_scatter(XT, count, 1000)  # Plotting 1000th Image

        plt.savefig("Images/Sample Plot Series/Gaussian/E " + str(self.epochCounter) + " T " + str(dataIndex) + ".jpg")
        plt.close()

    def plot_scatter(self, XT, count, title):
        plt.subplot(1, self.sampleCount + 1, int(self.sampleCount + 1 - (count / self.imageSeries)))
        plt.xlim(-1_000, 1_000)
        plt.ylim(-1_000, 1_000)
        plt.title("T " + str(title))
        for i in range(0, self.batchSize):
            plt.scatter((XT[i][:, 0].cpu()), (XT[i][:, 1].cpu()), c="b", linewidths=1)


    def gradient_descent(self, model, idx, X0, t, MI):  # TRAINING 1: Data = 2: X_0 ∼ q(x_0)
        XT, epsilon = self.q_sample(X0, idx, t)
        predictedNoise = model(XT, t)

        # MSE(Input, Target) | UNet(XT, t)
        # F.nll_loss only 3D supported, gets 4D
        loss = F.mse_loss(XT, predictedNoise) # TRAINING 2: Loss = MSE(XT, UNet(XT, t))
        return loss, XT, epsilon, predictedNoise # TRAINING 3: Backpropagate Loss


    def train_and_sample(self, model, dataset, MI):
        flag = False
        # MI in here somewhere
        for idx, data in enumerate(dataset):  # TRAINING 1: Repeated Loop
            self.optimizer.zero_grad()

            t = torch.randint(0, self.t_steps, (data.shape[0],), device=device).long()  # TRAINING 3: t ∼ Uniform({1, . . . , T })
            self.loss, XT, epsilon, predNoise = self.gradient_descent(model, idx, data.to(device), t, MI) # TRAINING 4: Loss = ||  − _θ * (√ (̄α_t) * x_0 + √(1−α_t) * , t) || ^ 2

            if int(idx % (len(dataset)/5)) == 0:
                print(f"T: {idx:05d}/{len(dataset)}:\tLoss: {self.loss.item()}")

            # Conduct intermittent sampling during training process to demonstrate progress

            if (((self.epochCounter % round(self.epochs * (1/5)) == 0) or (self.epochCounter == self.epochs)) and flag == False and (self.dimensionality == 2)):
                #self.imgGenFromSamples(model, img, "Images/Generated Images/E " + str(self.epochCounter) + " T " + str(int(idx)) + ".jpg")
                self.plot_2D_gaussian(model, XT, idx)  # Plot Sample Transformation over time
                flag = True

            self.loss.backward()
            self.optimizer.step()
        print(f"T: {int((self.sampleSize/self.batchSize)+1):05d}/{len(dataset)}:\tLoss: {self.loss.item()}") # Print final loss

    def run(self):

        # Get Data
        data = self.getDataset()
        normalized_data = (data - data.min())/(data.max() - data.min()) * 2 - 1  # Normalize between [-1, 1]
        dataset = DataLoader(dataset=normalized_data, batch_size=self.batchSize, shuffle=True, pin_memory=True)

        # Get Models
        model = self.getModel()
        MI = torch.load("ddpm models/MINeuralEstimator.pt")

        # Train and Sample
        print("Start Time: " + str(datetime.datetime.now()) + "")
        while self.epochCounter != self.epochs+1:
            print(f"\n   ------------- Epoch {self.epochCounter} ------------- ")
            self.train_and_sample(model, dataset, MI)  # Sampling done intermittently during training
            self.epochCounter += 1

        torch.save(model, f"ddpm models/{self.dimensionality}D_Gauss_ddpm_E{self.epochCounter - 1}_L{round(self.loss.item(), 5)}.pt") # Save Model

        print(f"\n   ------------- Epoch {self.epochCounter - 1} ------------- ")
        print(f"\t  Final Loss: {self.loss.item()}\n")

        # System Information
        print(f'GPU utilization:\t {deviceID.gpu_utilization()}%')
        print(f'Total memory:\t    {deviceID.memory_total_human()}')
        print(f'Used memory:\t     {deviceID.memory_used_human()}')
        print(f'Free memory:\t     {deviceID.memory_free_human()}')

        print("Completion Time: " + str(datetime.datetime.now()) + "")

if __name__ == "__main__":
    main()