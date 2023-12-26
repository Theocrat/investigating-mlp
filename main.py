import tqdm
import torch

from loading import Iris
from model import IrisClassifier
from matplotlib import pyplot as plt

# Settings
NUM_EPOCHS = 100
BATCH_SIZE = 10

# Data, Model, Loss, Optimizer
data = torch.utils.data.DataLoader(Iris(), batch_size=BATCH_SIZE)
model = IrisClassifier()
adam = torch.optim.Adam(model.parameters(), lr=0.1)
cross_entropy = torch.nn.CrossEntropyLoss()

# Training loop
losses = []
for n in (bar := tqdm.tqdm(range(NUM_EPOCHS))):
    epoch_losses = []
    
    adam.zero_grad()
    for X, d in data:
        y = model(X)

        loss = cross_entropy(y, d)
        loss.backward()
        epoch_losses.append(loss.item())
    
    adam.step()
    mean_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(mean_loss)
    bar.set_description(f"Loss: {round(mean_loss, 3)}")

    # Weight heatmap
    weights = model.l1_weights
    max_weight = round(float(max(weights.ravel())), 3)
    min_weight = round(float(min(weights.ravel())), 3)

    plt.clf()
    plt.imshow(weights, cmap="hot")
    plt.xlabel(f"Epoch {n + 1} of {NUM_EPOCHS}")
    plt.ylabel(f"max: {max_weight}, min: {min_weight}")
    plt.savefig(f"weights/{n + 1}.png")

    # Gradient heatmap
    grads = model.l1_wgrads
    max_grad = round(float(max(grads.ravel())), 3)
    min_grad = round(float(min(grads.ravel())), 3)

    plt.clf()
    plt.imshow(grads, cmap="Blues")
    plt.xlabel(f"Epoch {n + 1} of {NUM_EPOCHS}")
    plt.ylabel(f"max: {max_grad}, min: {min_grad}")
    plt.savefig(f"grads/{n + 1}.png")


# Plotting the training output
plt.figure(figsize=(20, 10))
plt.clf()
plt.grid()
plt.plot(range(NUM_EPOCHS), losses, color="#d46")
plt.fill_between(range(NUM_EPOCHS), [0] * NUM_EPOCHS, losses, color="#d463")
plt.tight_layout()
plt.savefig("losses.png")