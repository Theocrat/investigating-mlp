import tqdm
import json
import torch

from loading import Iris
from model import IrisClassifier
from matplotlib import pyplot as plt

# Settings
NUM_EPOCHS = 100
BATCH_SIZE = 10
LEARNING_RATE = 0.09

# Data, Model, Loss, Optimizer
data = torch.utils.data.DataLoader(Iris(), batch_size=BATCH_SIZE)
model = IrisClassifier()
cross_entropy = torch.nn.CrossEntropyLoss()

adam = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
sgd = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
optimizer = adam

# Training loop
losses = []
weights = {}
grads = {}
for n in (bar := tqdm.tqdm(range(NUM_EPOCHS))):
    epoch_losses = []
    
    optimizer.zero_grad()
    for X, d in data:
        y = model(X)

        loss = cross_entropy(y, d)
        loss.backward()
        epoch_losses.append(loss.item())
    
    optimizer.step()
    mean_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(mean_loss)
    bar.set_description(f"Loss: {round(mean_loss, 3)}")

    # Weight heatmap
    current_weights = model.l1_weights
    weights[n + 1] = {
        "nrows": current_weights.shape[0],
        "ncols": current_weights.shape[1],
        "data": [list([float(cell) for cell in row]) for row in current_weights]
    }

    # Gradient heatmap
    current_grads = model.l1_wgrads
    grads[n + 1] = {
        "nrows": current_grads.shape[0],
        "ncols": current_grads.shape[1],
        "data": [list([float(cell) for cell in row]) for row in current_grads]
    }


# Plotting the training output
plt.figure(figsize=(20, 10))
plt.clf()
plt.grid()
plt.plot(range(NUM_EPOCHS), losses, color="#d46")
plt.fill_between(range(NUM_EPOCHS), [0] * NUM_EPOCHS, losses, color="#d463")
plt.tight_layout()
plt.savefig("losses.png")

with open("weights/l1.js", "w") as f:
    print("modelWeightsL1 =", json.dumps(weights, indent=2), file=f)

with open("grads/l1.js", "w") as f:
    print("modelGradsL1 =", json.dumps(grads, indent=2), file=f)