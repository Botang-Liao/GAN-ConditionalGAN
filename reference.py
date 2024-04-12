import numpy as np
import torch
from config import DEVICE
from model import ConditionalGAN
from torchvision.utils import make_grid
import matplotlib.pylab as plt
from torch.autograd import Variable


def plot_all_images(generator:ConditionalGAN, label_names, picture_name='all_images.png'):
    generator_random_input = Variable(torch.randn(100, 100)).to(DEVICE)
    labels = Variable(torch.LongTensor([i for _ in range(10) for i in range(10)])).to(DEVICE)
    sample_images = generator(generator_random_input, labels).unsqueeze(1).data.cpu()
    grid = make_grid(sample_images, nrow=10, normalize=True).permute(1,2,0).numpy()
    fig, ax = plt.subplots(figsize=(15,15))
    ax.imshow(grid)
    plt.yticks([])
    plt.xticks(np.arange(15, 300, 30), label_names, rotation=45, fontsize=20)
    plt.savefig('./result/'+ picture_name) # Save the combined figure
    plt.close()
    print('All images plotted successfully.')
    

def plot_a_image(generator:ConditionalGAN, picture_name):
    generator_random_input = Variable(torch.randn(9, 100)).to(DEVICE)
    labels = Variable(torch.LongTensor(np.arange(9))).to(DEVICE)
    sample_images = generator(generator_random_input, labels).unsqueeze(1).data.cpu()
    grid = make_grid(sample_images, nrow=3, normalize=True).permute(1,2,0).numpy()
    plt.imshow(grid)
    plt.savefig('./result/'+ picture_name) # Save the combined figure
    plt.close()
        
    


if __name__ == '__main__':
    pass