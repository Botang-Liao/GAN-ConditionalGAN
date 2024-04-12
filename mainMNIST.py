from model import ConditionalGAN
from conditionalGANDatasetManger import MNISTDatasetManager
from train import Trainer
from lossFunction import BinaryCrossEntropyLoss

if __name__ == '__main__':
    
    model = ConditionalGAN()
    model.load_models()
    data_loader = MNISTDatasetManager()
    loss_function = BinaryCrossEntropyLoss()
    
    trainer = Trainer(model, data_loader, loss_function)
    trainer.train()