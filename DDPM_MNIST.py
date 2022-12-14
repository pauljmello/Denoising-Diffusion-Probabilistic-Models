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

SEED = 1001
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
    epochs = 10
    lr = 2e-4
    batchSize = 48  # Valid Batch Size divisible by 60k, [8, 16, 32, 48, 96, 125, 250]
    schedule = "linear"  # "linear", "cosine" https://arxiv.org/abs/2102.09672

    sampleCount = 10

    steps = 1000
    t_steps = 1000

    DDPM = DDPM_MNIST(steps, t_steps, epochs, batchSize, lr, sampleCount, schedule)
    DDPM.run()


class DDPM_MNIST:

    def __init__(self, steps, t_steps, epochs, batchSize, lr, sampleCount, schedule):
        super().__init__()

        self.lr = lr
        self.steps = steps
        self.epochs = epochs
        self.t_steps = t_steps
        self.batchSize = batchSize
        self.sampleCount = sampleCount

        self.SaveSample = True  # X_hat_0 from Sampling
        self.SaveInputImage = False  # Image Input for Training
        self.sampleInterval = int(((60000 / self.batchSize) / 2))  # How often we generate images to view
        self.imageSeries = int(self.t_steps / (self.sampleCount))

        self.Debug = False
        self.minimalData = False  # Reduce Dataset for faster debug
        self.newSubSet = False  # Keep False Unless Necessary, get new subset for minimal training data each epoch
        self.minDataSize = 1000  # When self.minimalData = True, we define data size here

        self.loss = 0.0
        self.epochCounter = 0

        self.CollectGraphs = True  # Collect Data for Graphs
        self.lossList = []
        self.UNet_XTX0List = []
        self.UNet_X0XTList = []
        self.UNet_X0X0List = []
        self.e_UNetX0List = []

        self.Beta = self.getSchedule(schedule)  # Schedule
        self.Sqrd_Sigma = self.Beta  # Sigma^2
        self.Alpha = 1.0 - self.Beta  # Alpha

        self.Alpha_Cumprod = torch.cumprod(self.Alpha, dim=0)  # Product Value
        self.Sqrd_Alpha_Cumprod = torch.sqrt(self.Alpha_Cumprod)
        self.Alpha_Cumprod_Previous = F.pad(self.Alpha_Cumprod[:-1], (1, 0), value=1.0)
        self.Sqrd_1_Minus_Alpha_Cumprod = torch.sqrt(1.0 - self.Alpha_Cumprod)
        self.Log_one_minus_Alpha_Cumprod = torch.log(1.0 - self.Alpha_Cumprod)

        self.Sqrd_Recipricol_Alpha_Cumprod = torch.sqrt(1 / self.Alpha_Cumprod)
        self.Sqrd_Recipricol_Alpha_Cumprod_Minus_1 = torch.sqrt(1 / self.Alpha_Cumprod-1)

        # q(x_{t - 1} | x_t, x_0)
        self.Posterior_Variance = self.Beta * (1.0 - self.Alpha_Cumprod_Previous) / (1.0 - self.Alpha_Cumprod)
        self.Posterior_Log_Clamp = np.log(np.maximum(self.Posterior_Variance, 1e-20))
        self.Posterior1 = (self.Beta * torch.sqrt(self.Alpha_Cumprod_Previous) / (1.0 - self.Alpha_Cumprod))
        self.Posterior2 = ((1.0 - self.Alpha_Cumprod_Previous) * torch.sqrt(self.Alpha) / (1.0 - self.Alpha_Cumprod))

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
        model = UNet(in_channels=1, model_channels=96, out_channels=1,
                     channel_mult=(1, 2, 4), attention_resolutions=[]).to(device)
        model = nn.DataParallel(model)  # Parallelize Data when Multi-GPU Applicable (Untested)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, eps=1e-8)
        return model

    def getDataset(self):
        dataTransformation = torchvision.transforms.Compose([
                                torchvision.transforms.Grayscale(num_output_channels=1),
                                torchvision.transforms.ToTensor(),
                                transforms.Lambda(lambda t: (t * 2) - 1)   # [-1, 1]
                                ])

        testData = torchvision.datasets.MNIST(root=".", download=True, train=False, transform=dataTransformation)
        trainData = torchvision.datasets.MNIST(root=".", download=True, train=True, transform=dataTransformation)

        if self.minimalData == True:
            # Subset of 1000 Images for Training and Sampling
            subset = list(np.random.choice(np.arange(0, len(trainData)), self.minDataSize, replace=False))
            DataSet = DataLoader(trainData, batch_size=self.batchSize, pin_memory=True, sampler=SubsetRandomSampler(subset))
        else:
            DataSet = DataLoader(trainData, batch_size=self.batchSize, pin_memory=True)

        return trainData, testData, DataSet


    def getExtract(self, tensor: torch.Tensor, t: torch.Tensor, X):
        out = tensor.gather(-1, t.cpu()).float()
        return out.reshape(t.shape[0], *((1,) * (len(X) - 1))).to(t.device)

    def q_mean_var(self, X0, t):
        X0_shape = X0.shape
        mean = self.getExtract(self.Sqrd_Alpha_Cumprod, t, X0_shape) * X0
        variance = self.getExtract(1.0 - self.Alpha_Cumprod, t, X0_shape)
        logVar = self.getExtract(self.Log_one_minus_Alpha_Cumprod, t, X0_shape)
        return mean, variance, logVar

    def q_posterior_mean_variance(self, X0, XT, t):
        XT_shape = XT.shape
        posterior_mean = self.getExtract(self.Posterior1, t, XT_shape) * X0 + self.getExtract(self.Posterior2, t, XT_shape) * XT
        posterior_var = self.getExtract(self.Posterior_Variance, t, XT_shape)
        posterior_log = self.getExtract(self.Posterior_Log_Clamp, t, XT_shape)
        return posterior_mean, posterior_var, posterior_log

    def q_sample(self, data, idx, t):  # Sample from Q(Xt | X0)
        epsilon = torch.randn_like(data)  #  ∼ N (0, I)

        # Mean, Variance LogVar from Q(Xt | X0)
        # mean, variance, logVariance = self.q_mean_var(data, t)
        # QSample = mean * data + variance.sqrt() * epsilon

        QSample = (self.getExtract(self.Sqrd_Alpha_Cumprod, t, data.shape) * data
                        + self.getExtract(self.Sqrd_1_Minus_Alpha_Cumprod, t, data.shape) * epsilon).float()

        if self.SaveInputImage == True and (idx == self.sampleInterval or idx == 0):
            plt.clf()
            # Examples of Corrupted Images used for Training
            self.save_image(QSample, "Images/Corrupted Images for Prediction/Example Input E "
                         + str(self.epochCounter) + " T " + str(idx) + ".jpg")

        return QSample, epsilon

    def pred_X0_from_XT(self, XT, t, epsilon):
        XT_shape = XT.shape
        return (self.getExtract(self.Sqrd_Recipricol_Alpha_Cumprod, t, XT_shape) * XT - self.getExtract(self.Sqrd_Recipricol_Alpha_Cumprod_Minus_1, t, XT_shape) * epsilon)

    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, XT, t):
        X0_prediction = self.pred_X0_from_XT(XT.float(), t, model(XT.float(), t))
        model_mean, posterior_Variance, posterior_log_variance = self.q_posterior_mean_variance(X0_prediction, XT, t)
        return model_mean, posterior_Variance, posterior_log_variance

    # Sampling
    @torch.no_grad()
    def p_sample(self, model, XT, t, tID):  # Getting Sample values at Time T for Processes Q & P
        model_mean, _, model_logVar = self.p_mean_variance(model, XT, t)
        epsilon = torch.randn_like(XT)
        # No noise at t == 0
        nonzeroMask = ((t != 0).float().view(-1, *([1] * (len(XT.shape) - 1))))
        sample = model_mean + nonzeroMask * torch.exp(0.5 * model_logVar) * epsilon
        return sample

    # Looped Sampling for Consistent Reverse Process
    @torch.no_grad()
    def sample(self, model, img):  # Sampling
        for count in range(self.t_steps):  # SAMPLING 2: for t = T, . . . , 1 do
            t = self.t_steps - count - 1
            # SAMPLING 3 & 4 See Backward PASS (Data = XT or X)
            img = self.p_sample(model, img, torch.full((self.batchSize,), t, device=device, dtype=torch.long), t)
        return img  # SAMPLING 6: return generated X_0 Data at End


    def save_image(self, img, msg):
        reverseProcess = transforms.Compose([
            transforms.Lambda(lambda t: (t * 2) - 1),  # [-1, 1]
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

    def generate_images(self, model, img, msg):
        plt.figure(figsize=(15, 15))
        X0 = self.sample(model, img)
        self.save_image(X0[0], msg)
        plt.close()

    def plot_graphs(self, arr, labelX, labelY, title, savePath):
        plt.plot(arr)
        plt.xlabel(labelX)
        plt.ylabel(labelY)
        plt.title(title)
        plt.savefig(savePath)
        plt.clf()

    @torch.no_grad()
    def plot_sample_quality(self, model, img, dataIndex):
        plt.figure(figsize=(15, 3))
        for count in range(0, self.steps)[::-1]:
            img = self.p_sample(model, img, torch.full((self.batchSize,), count, device=device, dtype=torch.long), count)
            if count == 0 or count % self.imageSeries == 0:
                plt.subplot(1, self.sampleCount, int(self.sampleCount - (count / self.imageSeries)))
                plt.title("T " + str(count))
                plt.imshow((img[0].cpu().squeeze().numpy() + 1.0) * 255 / 2, cmap="gray")

        plt.savefig("Images/Sample Plot Series/mnist/E " + str(self.epochCounter) + " T " + str(dataIndex) + ".jpg")
        plt.close()


    @torch.no_grad()
    def forward_diffusion_example(self, model, img):
        plt.figure(figsize=(15, 3))
        for count in range(self.steps):
            if count == 0 or count % self.imageSeries == 0:
                plt.subplot(1, self.sampleCount, int((count / self.imageSeries) + 1))
                plt.title("T " + str(count))
                plt.imshow((img[0].cpu().squeeze().numpy() + 1.0) * 255 / 2, cmap="gray")

            img = self.p_sample(model, img, torch.full((self.batchSize,), count, device=device, dtype=torch.long), count)

        plt.savefig("Images/Example Gradual Corruption.jpg")
        plt.close()

        return img


    def save_final_data(self, model, img):
        if self.CollectGraphs == True:
            self.plot_graphs(self.lossList, "Time", "Loss", "Training Loss", "Images/Graphs/Training Loss.jpg")
            self.plot_graphs(self.e_UNetX0List, "X", "Y", "e_UNetX0", "Images/Graphs/e_UNetX0.jpg")
            self.plot_graphs(self.UNet_XTX0List, "X", "Y", "UNet_XTX0", "Images/Graphs/UNet_XTX0.jpg")
            self.plot_graphs(self.UNet_X0XTList, "X", "Y", "UNet_X0XT", "Images/Graphs/UNet_X0XT.jpg")
            self.plot_graphs(self.UNet_X0X0List, "X", "Y", "UNet_X0X0", "Images/Graphs/UNet_X0X0.jpg")

        self.generate_images(model, img, "Images/Generated Images/A Final Sample " + str(self.epochCounter) + ".jpg")
        self.generate_images(model, img, "Images/Final Generated Image.jpg")

        self.plot_sample_quality(model, img, 00000)



    def gradient_descent(self, model, idx, X0, t, MI):  # TRAINING 1: Data = 2: X_0 ∼ q(x_0)

        XT, epsilon = self.q_sample(X0, idx, t)
        predictedNoise = model(XT, t)

        # MSE(Input, Target) | UNet(XT, t)
        # F.nll_loss only 3D supported, gets 4D
        loss = F.mse_loss(epsilon, predictedNoise)  # TRAINING 5: ∇θ ||  − _θ * (√ (̄α_t) * x_0 + √(1−α_t) * , t) || ^ 2
        if self.CollectGraphs == True and (idx == self.sampleInterval or idx == 0):
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

    def train_and_sample(self, model, img, DataSet, MI):

        # MI in here somewhere

        for idx, (X0, labels) in enumerate(DataSet):  # TRAINING 1: Repeated Loop
            self.optimizer.zero_grad()
            if self.Debug == True:
                print(np.array(X0))

            # Gradual Degradation Code Goes Here

            # TRAINING 3: t ∼ Uniform({1, . . . , T })
            t = torch.randint(0, self.t_steps, (X0.shape[0],), device=device).long()
            # TRAINING 2: X_0 ∼ q(x_0)
            self.loss, XT, epsilon, predNoise = self.gradient_descent(model, idx, X0.to(device), t, MI)

            if int(idx % (len(DataSet)/1000)) == 0:
                print(f"T: {idx:05d}/{len(DataSet)}:\tLoss: {self.loss.item()}")

            # Conduct intermittent sampling during training process to demonstrate progress
            plt.close()
            if self.SaveSample == True and (idx == 0 or idx == self.sampleInterval):
                self.generate_images(model, img, "Images/Generated Images/E " + str(self.epochCounter)
                                       + " T " + str(int(idx)) + ".jpg")
                self.plot_sample_quality(model, img, idx)  # Generate over time

            self.loss.backward()
            self.optimizer.step()
        print(f"T: {self.batchSize:05d}/{len(DataSet)}:\tLoss: {self.loss.item()}")

    def run(self):

        model = self.getModel()
        MI = torch.load("ddpm models/MINeuralEstimator.pt")
        trainData, testData, DataSet = self.getDataset()

        print("Start Time: " + str(datetime.datetime.now()) + "")

        image = next(iter(DataSet))[0][0][0]  # 28x28 Tensor for MNIST
        img = next(iter(DataSet))[0]  # 28x28 Tensor for MNIST
        visualizeInputImg = img.to(device)  # 28x28 Tensor for MNIST

        if self.Debug == True:  # Print Image to terminal in Float form
            print(np.array(image))

        self.save_image(visualizeInputImg, "Images/Example Input Image.jpg")

        # Example of Forward Diffusion Process, Same as in plotSampleQuality but not reversed
        noisy5 = self.forward_diffusion_example(model, img.to(device))

        plt.figure(figsize=(15, 15))
        self.save_image(visualizeInputImg.detach().cpu(), "Images/Generated Images/An Input Image.jpg")

        # epsilon = torch.randn(1, 1, img.shape[0], img.shape[0]).to(device)  # Example Noise Used

        while self.epochCounter != self.epochs:
            print(f"\n   ------------- Epoch {self.epochCounter} ------------- ")
            self.train_and_sample(model, visualizeInputImg, DataSet, MI)  # Sampling done intermittently during training

            # Regenerate New Samples Each Epoch Cycle
            if self.newSubSet == True and self.minimalData == True:
                subset = list(np.random.choice(np.arange(0, len(trainData)), self.minDataSize, replace=False))
                DataSet = DataLoader(trainData, batch_size=self.batchSize, pin_memory=True, sampler=SubsetRandomSampler(subset))

            self.epochCounter += 1

        if self.minimalData == True:
            torch.save(model, "ddpm models/minimal_ddpm.pt")
        else:
            torch.save(model, f"ddpm models/mnist_ddpm_E{self.epochCounter}_L{round(self.loss.item(), 5)}.pt")

        # Generate Data Plots
        self.save_final_data(model, visualizeInputImg)

        print(f"\n   ------------- Epoch {self.epochCounter} ------------- ")
        print(f"\t  Final Loss: {self.loss.item()}\n")

        print(f'GPU utilization:\t {deviceID.gpu_utilization()}%')
        print(f'Total memory:\t    {deviceID.memory_total_human()}')
        print(f'Used memory:\t     {deviceID.memory_used_human()}')
        print(f'Free memory:\t     {deviceID.memory_free_human()}')

        print("Completion Time: " + str(datetime.datetime.now()) + "")

if __name__ == "__main__":
    main()
