import torch
from torchvision import datasets, transforms
from config import BATCH_SIZE, MNIST_DATASET_PATH, FASHION_MINST_DATASET_PATH, MNIST_DATASET_LABELS, FASHION_MINST_DATASET_LABELS



class DatasetManager:
    def __init__(self, dataset_class, dataset_path, dataset_labels):
        self.training_data = self.data2dataloader(dataset_class, dataset_path)
        self.dataset_labels = dataset_labels
    def data2dataloader(self, dataset_class, dataset_path):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        dataset = dataset_class(dataset_path, train=True, download=True, 
                                transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset, 
                                                  batch_size=BATCH_SIZE, 
                                                  shuffle=True)
        return data_loader
        
class MNISTDatasetManager(DatasetManager):
    def __init__(self):
        super().__init__(datasets.MNIST, MNIST_DATASET_PATH, MNIST_DATASET_LABELS)
        
class FashionMNISTDatasetManager(DatasetManager):
    def __init__(self):
        super().__init__(datasets.FashionMNIST, FASHION_MINST_DATASET_PATH, FASHION_MINST_DATASET_LABELS)
