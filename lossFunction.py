import torch


class LossFunction:
    def __init__(self, criterion):
        self.criterion = criterion

    def discriminatorLoss(self, real, fake):
        real_loss = self.criterion(real, torch.ones_like(real))
        fake_loss = self.criterion(fake, torch.zeros_like(fake))
        return real_loss + fake_loss

    def generatorLoss(self, fake):
        return self.criterion(fake, torch.ones_like(fake))

class BinaryCrossEntropyLoss(LossFunction):
    def __init__(self):
        criterion = torch.nn.BCELoss()
        super().__init__(criterion)

class MeanSquaredErrorLoss(LossFunction):
    def __init__(self):
        criterion = torch.nn.MSELoss()
        super().__init__(criterion)
