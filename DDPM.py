# Author: Paul-Jason Mello
# Date: November 4th, 2022

# Used Libraries

import torch
import psutil  # Monitor GPU Memory Usage
import datetime
import torchvision
import torch.utils.data
import numpy as np
import torch.nn.functional as F

# Specific Torch
from torch import optim, nn
from torch.distributed import gather
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

# Misc. Libraries
from typing import Optional
from matplotlib import pyplot as plt

# Own Libraries
from DDPMUNet import DDPMUNet
from gather import gather

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    diffusionSteps = 1000
    sequenceSteps = 1000
    epochs = 5
    batchSize = 1
    sampleFreq = 1
    learningRate = 0.0002
    minLin = 0.0001
    maxLin = 0.02
    timeEmbedding = 128

    DDPM = DenoisingDiffusionProbabilisticModel(
        diffusionSteps, sequenceSteps, epochs, batchSize,
        sampleFreq, learningRate, minLin, maxLin, timeEmbedding
    )
    DDPM.run()


class DenoisingDiffusionProbabilisticModel:

    def __init__(self, steps, t_steps, epochs, batchSize, sampleFreq, lr, minRate, maxRate, timeEmbedding):
        super().__init__()

        self.steps = steps
        self.t_steps = t_steps
        self.epochs = epochs
        self.sampleFreq = sampleFreq
        self.batchSize = batchSize

        self.SaveInputImage = False  # Image Input for Training
        self.CollectGraphs = True  # Collect Data for Graphs
        self.minimalData = True  # Reduce Dataset for faster debug
        self.SaveSample = True  # X_hat_0 from Sampling
        self.newSubSet = False  # Keep False Unless Necessary, get new subset for minimal training data each epoch
        self.Debug = False

        self.imageSeries = int(self.steps / (10))
        self.sampleInterval = 500  # How often we generate images to view
        self.graphInterval = 250  # How often we collect datapoints to graph
        self.minDataSize = 1000  # When self.minimalData = True, we define data size here x < 10_000
        self.epochs = 1  # Quick Epoch Change

        self.epochCounter = 0
        self.loss = 0.0

        self.lossList = []
        self.UNet_XTX0List = []
        self.UNet_X0XTList = []
        self.UNet_X0X0List = []
        self.e_UNetX0List = []

        self.EpsilonUNet = DDPMUNet(self.t_steps, timeEmbedding).to(device)
        self.EpsilonUNet = nn.DataParallel(self.EpsilonUNet)  # Parallelize Data when Multi-GPU Applicable (Untested)
        self.optimizer = optim.Adam(self.EpsilonUNet.parameters(), lr=lr, eps=0.00000001)

        dataTransformation = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),  # Convert to Single Channel
            torchvision.transforms.ToTensor(),  # Convert to Tensor
            transforms.Lambda(lambda x: (x * 2) - 1)])  # Rescale Data to [-1, 1]

        self.testData = torchvision.datasets.MNIST(root=".", download=True, train=False, transform=dataTransformation)
        self.trainData = torchvision.datasets.MNIST(root=".", download=True, train=True, transform=dataTransformation)

        if self.minimalData == True:
            subset = list(np.random.choice(np.arange(0, len(self.trainData)), self.minDataSize, replace=False))  # Subset of 1000 Images for Training and Sampling
            self.MNISTData = DataLoader(self.trainData, batch_size=batchSize, pin_memory=True, sampler=SubsetRandomSampler(subset))
        else:
            self.MNISTData = DataLoader(self.trainData, batch_size=batchSize, pin_memory=True)

        # Mix Train and Test data together
        # self.MNISTData = DataLoader(torch.utils.data.ConcatDataset([self.trainData, self.testData]), batch_size = self.batchSize, pin_memory = True)

        self.Beta = torch.linspace(minRate, maxRate, self.t_steps)  # Linearized Variance Schedule

        self.Sqrd_Sigma = self.Beta     # Sigma^2
        self.Alpha = 1.0 - self.Beta  # Alpha

        self.Alpha_Cumprod = torch.cumprod(self.Alpha, dim=0)  # Product Value
        self.Sqrd_Alpha_Cumprod = torch.sqrt(self.Alpha_Cumprod)
        self.Alpha_Cumprod_Previous = np.append(1.0, self.Alpha_Cumprod[:-1])
        self.Sqrd_1_Minus_Alpha_Cumprod = torch.sqrt(1.0 - self.Alpha_Cumprod)

    def Q_Mean_Var(self, data, timeStep):  # Q(X_t|X_0)
        variance = 1 - gather(self.Alpha_Cumprod, timeStep)  # Variance
        mean = gather(self.Sqrd_Alpha_Cumprod, timeStep) * data  # Mean
        return variance, mean

    def Q_Sample(self, data, dataIndex, timeStep, epsilon: Optional[torch.Tensor] = None):  # Sample from Q(Xt | X0)
        if epsilon is None:
            epsilon = torch.randn(data.shape).to(device)  #  ∼ N (0, I)
        variance, mean = self.Q_Mean_Var(data, timeStep)  # Variance, Mean from Q(Xt | X0)

        QSample = mean + variance.sqrt() * epsilon

        if self.SaveInputImage == True and dataIndex % self.sampleInterval == 0:
            plt.clf()
            self.saveImg(QSample, "Images/Corrupted Images for Prediction/Example Input E." + str(self.epochCounter) + " T." + str(dataIndex) + ".jpg")  # Random Noise MNIST

        return QSample

    # Reverse Process Sampling
    def P_Sample(self, XT, timeStep):  # Getting Sample values at Time T for Processes Q & P
        epsilonTheta = self.EpsilonUNet(XT, timeStep)
        alpha_bar = gather(self.Alpha_Cumprod, timeStep)
        alpha = gather(self.Alpha, timeStep)
        epsilonCoef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** .5) * (XT.to(device) - epsilonCoef * epsilonTheta)
        variance = gather(self.Sqrd_Sigma, timeStep)
        epsilon = torch.randn(XT.shape, device=device)
        return mean + (variance ** .5) * epsilon

    # Sampling for Reverse Process
    @torch.no_grad()
    def Sample(self, XT):  # Sampling
        for numT in range(self.t_steps):  # SAMPLING 2: for t = T, . . . , 1 do
            timeVal = self.t_steps - numT - 1
            XT = self.P_Sample(XT, XT.new_full((self.sampleFreq,), timeVal, dtype=torch.long))  # SAMPLING 3 & 4 See Backward PASS (Data = XT or X)
        return XT  # SAMPLING 6: return X_0 Data at End

    def getIDX(self, data, timeVal, xData):
        val = data.gather(-1, timeVal.cpu())  # Skipping to final timestamp
        return val.reshape(timeVal.shape[0], *((1,) * (len(xData) - 1))).to(timeVal.device)

    def imageTransform(self, inputImage):
        reverseProcess = transforms.Compose([
            transforms.Lambda(lambda tensor: (tensor + 1) / 2),
            transforms.Lambda(lambda tensor: tensor.permute(1, 2, 0)),
            transforms.Lambda(lambda tensor: tensor * 255),
            transforms.Lambda(lambda tensor: tensor.cpu().detach().numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        if len(inputImage.shape) == 4:
            inputImage = inputImage[0, :, :, :]
        plt.imshow(reverseProcess(inputImage), cmap="gray")

    def saveImg(self, img, msg):
        plt.imshow(img.squeeze().cpu(), cmap="gray")
        plt.title(str("{:.6f}".format(self.loss)))
        plt.savefig(msg)

    def imgGenFromSamples(self, noise, msg):
        X0 = self.Sample(noise)
        self.saveImg(X0[0], msg)

    def plotGraphs(self, list, labelX, labelY, title, savePath):
        plt.plot(list)
        plt.xlabel(labelX)
        plt.ylabel(labelY)
        plt.title(title)
        plt.savefig(savePath)
        plt.clf()

    @torch.no_grad()
    def plotSampleQuality(self, img, dataIndex):

        for count in range(0, self.steps, self.imageSeries)[::-1]:
            t = torch.full((self.sampleFreq,), count, dtype=torch.long).to(device)
            plt.subplot(1, 10, int(10 - (count / self.imageSeries + 1) + 1))
            plt.title("T " + str(count))
            img = self.P_Sample(img, t)
            self.imageTransform(img.detach().cpu())

        plt.savefig("Images/Sample Plot Series/E." + str(self.epochCounter) + " T." + str(dataIndex) + ".jpg")
        plt.clf()

    def gradientDescent(self, dataIndex, X0, epsilon: Optional[torch.Tensor] = None):  # TRAINING 1: Data = 2: X_0 ∼ q(x_0)

        t_uniform = torch.randint(0, self.t_steps, (X0.shape[0], X0.shape[0])).to(device)  # TRAINING 3: t ∼ Uniform({1, . . . , T })
        #t_uniform = torch.randint(count, count + 1, (X0.shape[0], X0.shape[0])).to(device)  # TRAINING 3: Use for total corruption 1 - T timesteps
        XT = self.Q_Sample(X0, dataIndex, t_uniform, epsilon)

        # MSE(Input, Target) | UNet(XT, t)
        MSE = F.mse_loss(epsilon, self.EpsilonUNet(XT, t_uniform))  # TRAINING 5: ∇θ ||  − _θ * (√ (̄α_t) * x_0 + √(1−α_t) * , t) || ^ 2
        if self.CollectGraphs == True and dataIndex % self.graphInterval == 0:
            self.lossList.append(MSE.item())
            eps_X0_t = self.EpsilonUNet(X0, t_uniform)

            e_UNetX0 = F.mse_loss(epsilon, eps_X0_t)
            self.e_UNetX0List.append(e_UNetX0.item())

            UNet_X0X0 = F.mse_loss(eps_X0_t, X0)
            self.UNet_X0X0List.append(UNet_X0X0.item())

            UNet_X0XT = F.mse_loss(eps_X0_t, XT)
            self.UNet_X0XTList.append(UNet_X0XT.item())

            UNet_XTX0 = F.mse_loss(self.EpsilonUNet(XT, t_uniform), X0)
            self.UNet_XTX0List.append(UNet_XTX0.item())

        return MSE

    def Training_and_Sampling(self, img):
        for dataIndex, data in enumerate(self.MNISTData):  # TRAINING 1: Repeated Loop
            if self.Debug == True:
                print(np.array(data))

            # Gradual Degradation Code Goes Here

            # Train on New MNIST N(0,1) Data
            epsilon = torch.randn(data[0].shape).to(device)
            self.optimizer.zero_grad()
            self.loss = self.gradientDescent(dataIndex, data[0].to(device), epsilon)  # TRAINING 2: X_0 ∼ q(x_0)
            self.loss.backward()
            self.optimizer.step()

            if dataIndex % 10 == 0:
                print(f"T: {dataIndex:05d}/{len(self.MNISTData)}:\tLoss: {self.loss.item()}")

            # Sample From Constant (5 or 7) MNIST Data
            # Conduct intermittent sampling during training process to demonstrate progress
            if self.SaveSample == True and dataIndex % self.sampleInterval == 0:
                    self.imgGenFromSamples(img, "Images/Generated Images/E." + str(self.epochCounter) + " T." + str(int(dataIndex)) + ".jpg")
                    self.plotSampleQuality(img, dataIndex)  # Generate over time

    def run(self):
        print("Start Time: " + str(datetime.datetime.now()) + "")

        image = next(iter(self.MNISTData))[0][0][0]  # 28x28 Tensor for MNIST
        img = next(iter(self.MNISTData))[0]  # 28x28 Tensor for MNIST
        visualizeInputImg = img.to(device)  # 28x28 Tensor for MNIST

        if self.Debug == True:  # Print Image to terminal in Float form
            print(np.array(image))

        #epsilon = torch.randn(1, 1, img.shape[0], img.shape[0]).to(device)  # Example Noise Used
        #t_uniform = torch.randint(0, 1, (img.shape[0], img.shape[0])).to(device)  # Example Uniform Distribution N(0, 1)

        self.saveImg(image, "Images/Example Input Image.jpg")

        # Example of Forward Diffusion Process, Same as in plotSampleQuality but not reversed
        plt.figure(figsize=(15, 15))
        for count in range(0, self.steps, self.imageSeries):
            tensor = torch.Tensor([count]).type(torch.int64).to(device)
            plt.subplot(1, int(10), int((count / self.imageSeries + 1)))
            plt.title("T " + str(count))
            imgData = self.P_Sample(img.to(device), tensor)
            self.imageTransform(imgData)

        plt.savefig("Images/Example Gradual Corruption.jpg")
        plt.clf()

        self.saveImg(visualizeInputImg.detach().cpu(), "Images/Generated Images/An Input Image.jpg")

        while (self.epochCounter != self.epochs):
            print(f"\n   ------------- Epoch {self.epochCounter} ------------- ")
            self.Training_and_Sampling(img.to(device))  # Sampling done intermittently during training

            # Regenerate New Samples Each Epoch Cycle
            if self.newSubSet == True and self.minimalData == True:
                subset = list(np.random.choice(np.arange(0, len(self.trainData)), self.minDataSize, replace=False))
                self.MNISTData = DataLoader(self.trainData, batch_size=self.batchSize, pin_memory=True, sampler=SubsetRandomSampler(subset))

            self.epochCounter += 1

        if self.minimalData == True:
            torch.save(self.EpsilonUNet, "Models/Minimal DDPM.h5")
        else:
            torch.save(self.EpsilonUNet, "Models/Full DDPM.h5")

        # Generate Data Plots
        plt.clf()
        if self.CollectGraphs == True:
            self.plotGraphs(self.lossList, "Time", "Loss", "Training Loss", "Images/Graphs/Training Loss.jpg")
            self.plotGraphs(self.e_UNetX0List, "X", "Y", "e_UNetX0", "Images/Graphs/e_UNetX0.jpg")
            self.plotGraphs(self.UNet_XTX0List, "X", "Y", "UNet_XTX0", "Images/Graphs/UNet_XTX0.jpg")
            self.plotGraphs(self.UNet_X0XTList, "X", "Y", "UNet_X0XT", "Images/Graphs/UNet_X0XT.jpg")
            self.plotGraphs(self.UNet_X0X0List, "X", "Y", "UNet_X0X0", "Images/Graphs/UNet_X0X0.jpg")

        self.imgGenFromSamples(visualizeInputImg, "Images/Generated Images/A Final Sample " + str(self.epochCounter) + ".jpg")
        self.imgGenFromSamples(visualizeInputImg, "Images/Final Generated Image.jpg")

        self.plotSampleQuality(visualizeInputImg, 00000)


        print(f"\n   ------------- Epoch {self.epochCounter} ------------- ")
        print(f"\t  Final Loss: {self.loss.item()}")
        print("\nFinal GPU VRAM Usage:" + str(psutil.virtual_memory().percent))
        print("Final Swap Mem Usage:" + str(psutil.swap_memory().percent))

        print("Completion Time: " + str(datetime.datetime.now()) + "")


if __name__ == "__main__":
    main()