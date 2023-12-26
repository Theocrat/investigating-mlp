import torch
from torch.utils.data import Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Iris(Dataset):
    """ Dataset class for Scikit-Learn Iris dataset """

    def __init__(self):
        """ Initialize the dataset class with the data from Scikit-Learn """
        self.iris = load_iris()
        self.data = self.iris["data"]
        self.labels = self.iris["target"]

    def __len__(self):
        """ Return the length of the Iris dataset in Scikit-Learn """
        return len(self.labels)

    
    def __getitem__(self, idx):
        """ Get one random sample for training """
        item_data = torch.tensor(self.data[idx, :], dtype=torch.float32)
        item_label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return item_data.to("cuda"), item_label.to("cuda")


if __name__ == "__main__":
    from torch.utils.data import DataLoader, random_split
    
    train, test = random_split(Iris(), [0.8, 0.2])
    iris = DataLoader(train, shuffle=True, batch_size=10)

    for X, y in iris:
        print("Sample data batch:", X)
        print("Sample label batch:", y)
        break

    print("Training data size:", len(train))
    print("Validation data size:", len(test))