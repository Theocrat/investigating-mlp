import torch
import torch.nn as nn

class IrisClassifier(nn.Module):
    """ Sequential classifier for the Iris dataset """

    def __init__(self):
        """ Initialize in the manner required for Torch models """
        super().__init__()

        # Model layers
        self.layer_1 = nn.Linear(4, 7, device="cuda", dtype=torch.float32)
        self.layer_2 = nn.Linear(7, 5, device="cuda", dtype=torch.float32)
        self.layer_3 = nn.Linear(5, 3, device="cuda", dtype=torch.float32)

        # Non-linear items
        self.act = nn.ReLU()
        self.max = lambda vector: torch.softmax(vector, 1)

    def forward(self, x_in):
        """ Run the forward pass on this model """
        hl_1 = self.act( self.layer_1(x_in) )
        hl_2 = self.act( self.layer_2(hl_1) )
        y_op = self.max( self.layer_3(hl_2) )
        return y_op

    @property
    def l1_weights(self):
        params = self.layer_1.parameters()
        weights = next(params).detach().to("cpu")
        return weights.numpy()
    
    @property
    def l1_wgrads(self):
        params = self.layer_1.parameters()
        wgrads = next(params).grad.detach().to("cpu")
        return wgrads.numpy()

if __name__ == "__main__":
    model = IrisClassifier()
    
    print("Model:")
    print(model)

    print("Layer 1 weights:")
    print(model.l1_weights)