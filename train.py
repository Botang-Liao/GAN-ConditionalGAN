import torch
from torch.autograd import Variable
import numpy as np

from config import BATCH_SIZE, DEVICE, NUM_EPOCHES
from model import ConditionalGAN
from conditionalGANDatasetManger import DatasetManager
from lossFunction import LossFunction
from reference import plot_all_images, plot_a_image


class Trainer():
    def __init__(self, model:ConditionalGAN, dataloader:DatasetManager, loss_function:LossFunction)->None:
        self.model = model
        self.dataloader = dataloader.training_data
        self.labels = dataloader.dataset_labels
        self.loss_function = loss_function
        self.discriminator_loss = []
        self.generator_loss = []
       
    def trainDiscriminator(self, real_images, real_images_label, fake_images, fake_images_label):
        self.model.discriminator_optimizer.zero_grad()
        discriminator_output_real = self.model.discriminator(real_images, real_images_label)
        discriminator_output_fake = self.model.discriminator(fake_images.detach(), fake_images_label)
        discriminator_loss = self.loss_function.discriminatorLoss(discriminator_output_real, discriminator_output_fake)
        discriminator_loss.backward()
        self.model.discriminator_optimizer.step()
        return discriminator_loss.item()
        
    def trainGenerator(self, fake_images, fake_labels):
        self.model.generator_optimizer.zero_grad()
        discriminator_output_fake = self.model.discriminator(fake_images, fake_labels)
        generator_loss = self.loss_function.generatorLoss(discriminator_output_fake)
        generator_loss.backward()
        self.model.generator_optimizer.step()
        return generator_loss.item()
        
    def train(self):
        for epoch in range(NUM_EPOCHES):
            self.model.generator.train()
            self.model.discriminator.train()
            this_discriminator_loss = 0.0
            this_generator_loss = 0.0
            for i, (images, labels) in enumerate(self.dataloader):   
                # 獲取真實影像資訊 
                real_images = Variable(images).to(DEVICE)
                real_images_label = Variable(labels).to(DEVICE)
                # 製作假影像
                generator_random_input = Variable(torch.randn(BATCH_SIZE, 100)).to(DEVICE)
                fake_images_label = Variable(torch.LongTensor(np.random.randint(0, 10, BATCH_SIZE))).to(DEVICE)   # 隨機亂數 [1, 10]
                fake_images = self.model.generator(generator_random_input, fake_images_label)
                this_discriminator_loss += self.trainDiscriminator(real_images, real_images_label, fake_images, fake_images_label)
                this_generator_loss += self.trainGenerator(fake_images, fake_images_label)
            self.discriminator_loss.append(this_discriminator_loss)
            self.generator_loss.append(this_generator_loss)
            print(f'Epoch {epoch+1}/{NUM_EPOCHES} Discriminator Loss: {this_discriminator_loss} Generator Loss: {this_generator_loss}')
            plot_a_image(self.model.generator, f'epoch_{epoch}.png')
            self.model.save_models()
        plot_all_images(self.model.generator, self.labels, 'all_images.png')
            