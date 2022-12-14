import numpy as np
import matplotlib.pyplot as plt


buffer = 1000 # Length of Kalman Filter Run

#Initialize Starting/Constant Variables
initState = np.array([[0.1],[-0.05],[np.pi + 0.1],[0.01]], float)
identity4x4 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
identify2x2 = np.array(([[1,0],[0,1]]))
c_censor = np.array([[0,1,0,0],[0,0,0,1]])
mean4 = np.array([0,0,0,0])
mean2 = np.array([0,0])

Vd = 0.0321 * identify2x2
Wd = 0.0123 * identity4x4

Sigma = []
Gain = []
S_Approx = []
S = []
Y = []

M = 1
m = .3
b = .1
L = 1
I = .01

constantDivision = ((I * (M + m)) + (M * m * (L*L)))

#Linearised EOM
X1 = (-(I + (m * (L*L))) * b) / constantDivision
X2 = (((m*m) * 9.8 * (L*L))) / constantDivision
X3 = (-(m*L*b)) / constantDivision
X4 = -(m*9.8*L*(M+m)) / constantDivision

bigMatrix = np.array(([[0,1,0,0],
                    [0,X1,X2,0],
                    [0,0,0,1],
                    [0,X3,X4,0]]), float)

print(bigMatrix)

TVal = (identity4x4 + bigMatrix * 0.01) # 0.01 = Seconds

S_Approx.append(initState)
Sigma.append(0.001 * identity4x4)
S.append(initState)


# Following matrices wrapped in np.mat due to improper python matrix multiplication
def step():
    WVal = np.mat(np.random.multivariate_normal(mean4, Wd, size=1).T)
    VVal = np.mat(np.random.multivariate_normal(mean2, Vd, size=1).T)
    S.append(np.mat(TVal) * np.mat(S[K]) + WVal)
    y = (np.mat(c_censor) * np.mat(S[K+1])) + VVal
    return y

K = 1
while K != buffer :
    Sigma.append((Wd) +
                 (np.mat(TVal) * np.mat(Sigma[K-1]) * np.mat(TVal.T)) -
                 ((np.mat(TVal) * np.mat(Sigma[K-1]) * np.mat(c_censor.T)) *
                 ((Vd + np.mat(c_censor) * np.mat(Sigma[K-1]) *
                 np.mat(c_censor.T))**-1) *
                 (np.mat(c_censor) * np.mat(Sigma[K-1]) * np.mat(TVal.T))))
    K = K + 1

K = 0
while K != buffer-1 :
    Gain.append((np.mat(TVal) * np.mat(Sigma[K]) * np.mat(c_censor.T)) * (Vd + np.mat(c_censor) * np.mat(Sigma[K]) * np.mat(c_censor.T))**-1)
    S_Approx.append(np.mat(TVal) * np.mat(S_Approx[K-1]) + (np.mat(Gain[K]) * (step() - (np.mat(c_censor) * np.mat(S_Approx[K-1])))))
    K = K + 1


#Final State Results
print("\n\nIteration Number: ", K)
print("S: \n", S[K])
print("SHat: \n", S_Approx[K])
print("Sigma: \n", Sigma[K])
print("Gain: \n", Gain[K-1])


#Plotting Final State Results
def plotGraphs(List, List_Approx, val):
    plt.figure(figsize=(15, 5))
    for i in range(len(List)):
        plt.scatter(i, List[i][val, 0], color = 'green', s=3)
        plt.scatter(i, List_Approx[i][val, 0], color = 'red', s=3)
    plt.legend(["Observation", "Prediction"], loc = "lower right")
    plt.show()


plotGraphs(S, S_Approx, 1)
plotGraphs(S, S_Approx, 3)