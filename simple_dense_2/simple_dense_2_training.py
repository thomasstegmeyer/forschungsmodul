#quick programm to print training progress

import matplotlib.pyplot as plt

epochs = list(range(3,30))

trainingloss = [80450.22, 80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22,80450.22]
validationloss = [80359.58,80360.32,80355.75, 80356.64,80355.70,80356.64, 80356.47,80357.06,80358.75,80354.80,80355.37, 80362.74,80356.28,80354.26,80360.09,80358.12, 80361.23,80358.77,80357.66,80356.09,80356.40, 80358.31,80356.60,80359.36,80355.69,80356.27, 80359.79]

print(len(trainingloss))
print(len(validationloss))

plt.plot(epochs,trainingloss)
plt.plot(epochs,validationloss)
plt.legend(["training","validation"])
plt.xlabel("Epochs")
plt.ylabel("MSE loss")

plt.savefig("training_progress.png")