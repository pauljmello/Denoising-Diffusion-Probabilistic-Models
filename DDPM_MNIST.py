# Author: Paul-Jason Mello
# Date: November 4th, 2022

# Used Libraries
import torch
import datetime
import torchvision
import torch.utils.data
import numpy as np
import torch.nn.functional as F

# Specific Torch
from torch import optim, nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

# Misc. Libraries
from matplotlib import pyplot as plt
from nvitop import Device  # GPU Monitoring

# Model Libraries
from UNet import UNet

#from DDPMUNet import DDPMUNet

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
    epochs = 10
    batchSize = 32  # Valid Batch Size divisible by 60k, [8, 16, 32, 48, 96, 125, 250]

    sampleCount = 10
    learningRate = 0.0002

    steps = 1000
    t_steps = 1000

    DDPM = DDPM_MNIST(steps, t_steps, epochs, batchSize, learningRate, sampleCount)
    DDPM.run()


class DDPM_MNIST:

    def __init__(self, steps, t_steps, epochs, batchSize, lr, sampleCount):
        super().__init__()

        self.lr = lr
        self.steps = steps
        self.epochs = epochs
        self.t_steps = t_steps
        self.batchSize = batchSize
        self.sampleCount = sampleCount

        self.cosineSchedule = False
        self.SaveInputImage = False  # Image Input for Training
        self.CollectGraphs = True  # Collect Data for Graphs
        self.SaveSample = True  # X_hat_0 from Sampling

        self.minimalData = False  # Reduce Dataset for faster debug
        self.newSubSet = False  # Keep False Unless Necessary, get new subset for minimal training data each epoch
        self.Debug = False

        self.sampleInterval = int(((60000 / self.batchSize) / 2))  # How often we generate images to view
        self.imageSeries = int(self.t_steps / (self.sampleCount))
        self.minDataSize = 1000  # When self.minimalData = True, we define data size here

        self.epochCounter = 0
        self.loss = 0.0

        self.lossList = []
        self.UNet_XTX0List = []
        self.UNet_X0XTList = []
        self.UNet_X0X0List = []
        self.e_UNetX0List = []

        if self.cosineSchedule is True:
            x = torch.linspace(0, self.t_steps, self.steps+1)
            y = torch.cos(((x/self.t_steps) + 0.01) / (1+0.01) * torch.pi*0.5) ** 2
            z = y / y[0]
            self.Beta = torch.clip((1 - (z[1:] / z[:-1])), 0.0001, 0.9999)
        else:
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
        # model = DDPMUNet(self.t_steps, self.timeEmbedding).to(device)
        model = UNet(in_channels=1, model_channels=96, out_channels=1,
                     channel_mult=(1, 2, 4), attention_resolutions=[]).to(device)
        model = nn.DataParallel(model)  # Parallelize Data when Multi-GPU Applicable (Untested)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, eps=0.00000001)
        return model

    def getDataset(self):
        dataTransformation = torchvision.transforms.Compose([
                                torchvision.transforms.Resize(28),
                                torchvision.transforms.Grayscale(num_output_channels=1),
                                torchvision.transforms.ToTensor(),
                                #transforms.Lambda(lambda t: (t * 2) - 1) #[-1,1]
                                ])

        testData = torchvision.datasets.MNIST(root=".", download=True, train=False, transform=dataTransformation)
        trainData = torchvision.datasets.MNIST(root=".", download=True, train=True, transform=dataTransformation)

        if self.minimalData is True:
            # Subset of 1000 Images for Training and Sampling
            subset = list(np.random.choice(np.arange(0, len(trainData)), self.minDataSize, replace=False))
            DataSet = DataLoader(trainData, batch_size=self.batchSize, pin_memory=True, sampler=SubsetRandomSampler(subset))
        else:
            DataSet = DataLoader(trainData, batch_size=self.batchSize, pin_memory=True)

        return trainData, testData, DataSet


    def getExtract(self, tensor: torch.Tensor, t: torch.Tensor, X):
        out = tensor.gather(-1, t.cpu()).float()
        return out.reshape(t.shape[0], *((1,) * (len(X) - 1))).to(t.device)

    def Q_Mean_Variance(self, X, t):
        mean = self.getExtract(self.Sqrd_Alpha_Cumprod, t, X.shape) * X
        variance = self.getExtract(self.Sqrd_1_Minus_Alpha_Cumprod, t, X.shape)
        logVar = self.getExtract(self.Log_one_minus_Alpha_Cumprod, t, X.shape)
        return mean, variance, logVar

    def Q_Posterior_Mean_Variance(self, X0, XT, t):
        pmean = (self.getExtract(self.Posterior1, t, XT.shape) * X0 + self.getExtract(self.Posterior2, t, XT.shape) * XT)
        pvar = self.getExtract(self.Posterior_Variance, t, XT.shape)
        plog = self.getExtract(self.Posterior_Log_Clamp, t, XT.shape)
        return pmean, pvar, plog

    def predictX0fromXT(self, XT, t, epsilon):
        return (self.getExtract(np.sqrt(1.0 / self.Alpha_Cumprod), t, XT.shape) *
                XT - self.getExtract(np.sqrt(1.0 / self.Alpha_Cumprod - 1), t, XT.shape) * epsilon)

    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def P_Mean_Variance(self, model, XT, t):
        X0 = self.predictX0fromXT(XT.float(), t, model(XT.float(), t))
        X0 = torch.clip(X0, min=-1., max=1.)
        modelMean, pvar, plog = self.Q_Posterior_Mean_Variance(X0, XT, t)
        return modelMean, pvar, plog

    def Q_Sample(self, data, idx, t):  # Sample from Q(Xt | X0)
        epsilon = torch.randn_like(data)  #  ∼ N (0, I)

        # Mean, Variance LogVar from Q(Xt | X0)
        mean, variance, logVariance = self.Q_Mean_Variance(data, t)

        QSample = mean * data + variance.sqrt() * epsilon

        if self.SaveInputImage is True and (idx == self.sampleInterval or idx is 0):
            plt.clf()
            # Random Noise MNIST
            self.saveImg(QSample, "Images/Corrupted Images for Prediction/Example Input E "
                         + str(self.epochCounter) + " T " + str(idx) + ".jpg")

        return QSample, epsilon

    # Sampling
    @torch.no_grad()
    def P_Sample(self, model, XT, t, tID):  # Getting Sample values at Time T for Processes Q & P
        mean, _, logVar = self.P_Mean_Variance(model, XT, t)
        noise = torch.randn_like(XT)
        mask = ((t != 0).float().view(-1, *([1] * (len(XT.shape) - 1))))
        XHat = mean + mask * (0.5 * logVar).exp() * noise
        return XHat

    # Looped Sampling for Consistent Reverse Process
    @torch.no_grad()
    def Sample(self, model, img):  # Sampling
        for count in range(self.t_steps):  # SAMPLING 2: for t = T, . . . , 1 do
            t = self.t_steps - count - 1
            # SAMPLING 3 & 4 See Backward PASS (Data = XT or X)
            img = self.P_Sample(model, img, torch.full((self.batchSize,), t, device=device, dtype=torch.long), t)
        return img  # SAMPLING 6: return generated X_0 Data at End


    def saveImg(self, img, msg):
        reverseProcess = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),  # [0,1]
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.cpu().detach().numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        if len(img.shape) == 4:
            img = img[0, :, :, :]

        plt.imshow(reverseProcess(img), cmap="gray")
        plt.title(str("{:.6f}".format(self.loss)))
        plt.savefig(msg)

    def imgGenFromSamples(self, model, img, msg):
        plt.figure(figsize=(15, 15))
        X0 = self.Sample(model, img)
        self.saveImg(X0[0], msg)
        plt.close()

    def plotGraphs(self, arr, labelX, labelY, title, savePath):
        plt.plot(arr)
        plt.xlabel(labelX)
        plt.ylabel(labelY)
        plt.title(title)
        plt.savefig(savePath)
        plt.clf()

    @torch.no_grad()
    def plotSampleQuality(self, model, img, dataIndex):
        plt.figure(figsize=(15, 3))
        for count in range(0, self.steps)[::-1]:
            img = self.P_Sample(model, img, torch.full((self.batchSize,), count, device=device, dtype=torch.long), count)
            if count == 0 or count % self.imageSeries == 0:
                plt.subplot(1, self.sampleCount, int(self.sampleCount - (count / self.imageSeries)))
                plt.title("T " + str(count))
                plt.imshow((img[0].cpu().squeeze().numpy() + 1.0) * 255 / 2, cmap="gray")

        plt.savefig("Images/Sample Plot Series/E " + str(self.epochCounter) + " T " + str(dataIndex) + ".jpg")
        plt.close()


    @torch.no_grad()
    def forwardExample(self, model, img):
        plt.figure(figsize=(15, 3))
        for count in range(self.steps):
            if count is 0 or count % self.imageSeries is 0:
                plt.subplot(1, self.sampleCount, int((count / self.imageSeries) + 1))
                plt.title("T " + str(count))
                plt.imshow((img[0].cpu().squeeze().numpy() + 1.0) * 255 / 2, cmap="gray")

            img = self.P_Sample(model, img, torch.full((self.batchSize,), count, device=device, dtype=torch.long), count)

        plt.savefig("Images/Example Gradual Corruption.jpg")
        plt.close()

        return img


    def saveFinalData(self, model, img):
        if self.CollectGraphs is True:
            self.plotGraphs(self.lossList, "Time", "Loss", "Training Loss", "Images/Graphs/Training Loss.jpg")
            self.plotGraphs(self.e_UNetX0List, "X", "Y", "e_UNetX0", "Images/Graphs/e_UNetX0.jpg")
            self.plotGraphs(self.UNet_XTX0List, "X", "Y", "UNet_XTX0", "Images/Graphs/UNet_XTX0.jpg")
            self.plotGraphs(self.UNet_X0XTList, "X", "Y", "UNet_X0XT", "Images/Graphs/UNet_X0XT.jpg")
            self.plotGraphs(self.UNet_X0X0List, "X", "Y", "UNet_X0X0", "Images/Graphs/UNet_X0X0.jpg")

        self.imgGenFromSamples(model, img, "Images/Generated Images/A Final Sample " + str(self.epochCounter) + ".jpg")
        self.imgGenFromSamples(model, img, "Images/Final Generated Image.jpg")

        self.plotSampleQuality(model, img, 00000)



    def gradientDescent(self, model, idx, X0, t, MI):  # TRAINING 1: Data = 2: X_0 ∼ q(x_0)

        XT, epsilon = self.Q_Sample(X0, idx, t)
        predictedNoise = model(XT, t)

        # MSE(Input, Target) | UNet(XT, t)
        # F.nll_loss only 3D supported, gets 4D
        loss = F.mse_loss(epsilon, predictedNoise)  # TRAINING 5: ∇θ ||  − _θ * (√ (̄α_t) * x_0 + √(1−α_t) * , t) || ^ 2
        if self.CollectGraphs is True and (idx == self.sampleInterval or idx is 0):
            self.lossList.append(loss.item())
            eps_X0_t = model(X0, t)

            e_UNetX0 = F.mse_loss(epsilon, eps_X0_t)
            self.e_UNetX0List.append(e_UNetX0.item())

            UNet_X0X0 = F.mse_loss(eps_X0_t, X0)
            self.UNet_X0X0List.append(UNet_X0X0.item())

            UNet_X0XT = F.mse_loss(eps_X0_t, XT)
            self.UNet_X0XTList.append(UNet_X0XT.item())

            UNet_XTX0 = F.mse_loss(model(XT, t), X0)
            self.UNet_XTX0List.append(UNet_XTX0.item())

        return loss, XT, epsilon, predictedNoise

    def TrainingAndSampling(self, model, img, DataSet, MI):

        # MI in here somewhere

        for idx, (X0, labels) in enumerate(DataSet):  # TRAINING 1: Repeated Loop
            self.optimizer.zero_grad()
            if self.Debug is True:
                print(np.array(X0))

            # Gradual Degradation Code Goes Here

            # TRAINING 3: t ∼ Uniform({1, . . . , T })
            t = torch.randint(0, self.t_steps, (X0.shape[0],), device=device).long()
            # TRAINING 2: X_0 ∼ q(x_0)
            self.loss, XT, epsilon, predNoise = self.gradientDescent(model, idx, X0.to(device), t, MI)
            #TODO predNoise in place of XT on plotSampleQuality

            if int(idx % (len(DataSet)/1000)) == 0:
                print(f"T: {idx:05d}/{len(DataSet)}:\tLoss: {self.loss.item()}")

            # Conduct intermittent sampling during training process to demonstrate progress
            plt.close()
            if self.SaveSample is True and (idx == 0 or idx is self.sampleInterval):
                self.imgGenFromSamples(model, img, "Images/Generated Images/E " + str(self.epochCounter)
                                       + " T " + str(int(idx)) + ".jpg")
                self.plotSampleQuality(model, img, idx)  # Generate over time

            self.loss.backward()
            self.optimizer.step()
        print(f"T: {self.batchSize:05d}/{len(DataSet)}:\tLoss: {self.loss.item()}")

    def run(self):

        model = self.getModel()
        MI = torch.load("DDPMModels/MINeuralEstimator.pt")
        trainData, testData, DataSet = self.getDataset()

        print("Start Time: " + str(datetime.datetime.now()) + "")

        image = next(iter(DataSet))[0][0][0]  # 28x28 Tensor for MNIST
        img = next(iter(DataSet))[0]  # 28x28 Tensor for MNIST
        visualizeInputImg = img.to(device)  # 28x28 Tensor for MNIST

        if self.Debug is True:  # Print Image to terminal in Float form
            print(np.array(image))

        self.saveImg(visualizeInputImg, "Images/Example Input Image.jpg")

        # Example of Forward Diffusion Process, Same as in plotSampleQuality but not reversed
        noisy5 = self.forwardExample(model, img.to(device))

        plt.figure(figsize=(15, 15))
        self.saveImg(visualizeInputImg.detach().cpu(), "Images/Generated Images/An Input Image.jpg")

        # epsilon = torch.randn(1, 1, img.shape[0], img.shape[0]).to(device)  # Example Noise Used

        while self.epochCounter != self.epochs:
            print(f"\n   ------------- Epoch {self.epochCounter} ------------- ")
            self.TrainingAndSampling(model, visualizeInputImg, DataSet, MI)  # Sampling done intermittently during training

            # Regenerate New Samples Each Epoch Cycle
            if self.newSubSet is True and self.minimalData is True:
                subset = list(np.random.choice(np.arange(0, len(trainData)), self.minDataSize, replace=False))
                DataSet = DataLoader(trainData, batch_size=self.batchSize, pin_memory=True, sampler=SubsetRandomSampler(subset))

            self.epochCounter += 1

        if self.minimalData is True:
            torch.save(model, "DDPMModels/Minimal DDPM.pt")
        else:
            torch.save(model, "DDPMModels/E10 Full DDPM.pt")

        # Generate Data Plots
        self.saveFinalData(model, visualizeInputImg)

        print(f"\n   ------------- Epoch {self.epochCounter} ------------- ")
        print(f"\t  Final Loss: {self.loss.item()}\n")

        print(f'GPU utilization:\t {deviceID.gpu_utilization()}%')
        print(f'Total memory:\t    {deviceID.memory_total_human()}')
        print(f'Used memory:\t     {deviceID.memory_used_human()}')
        print(f'Free memory:\t     {deviceID.memory_free_human()}')

        print("Completion Time: " + str(datetime.datetime.now()) + "")

if __name__ == "__main__":
    main()
