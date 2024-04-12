import torch

BATCH_SIZE = 64  # 批量

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 0.0002

MNIST_DATASET_PATH = '/HDD/n66104571/MNIST_data'
FASHION_MINST_DATASET_PATH = '/HDD/n66104571/FashionMNIST_data'

MODEL_NAME = 'basic_training100'

NUM_EPOCHES = 100

FASHION_MINST_DATASET_LABELS = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

MNIST_DATASET_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']