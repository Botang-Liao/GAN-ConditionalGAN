from torch import nn
import torch
from config import DEVICE, LR, MODEL_NAME

class ConditionalGAN():
    def __init__(self) -> None:
        self.generator = Generator().to(DEVICE)
        self.generator.apply(weights_init)
        self.discriminator = Discriminator().to(DEVICE)
        self.discriminator.apply(weights_init)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(),lr=LR)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=LR)
        
    def save_models(self):
        torch.save(self.generator.state_dict(), './model/'+ MODEL_NAME + 'generator.pth')
        torch.save(self.discriminator.state_dict(),'./model/'+ MODEL_NAME + 'discriminator.pth')
    
    def load_models(self):
        self.generator.load_state_dict(torch.load('./model/'+ MODEL_NAME + 'generator.pth'))
        self.discriminator.load_state_dict(torch.load('./model/'+ MODEL_NAME + 'discriminator.pth'))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02) # 卷積層權重初始值 
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02) # Batch Normalization 層權重初始值 
        m.bias.data.fill_(0)
        
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 設定嵌入層，作為 Label 的輸入
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)  # 合併輸入
        
        out = self.model(x)
        return out.squeeze()
    

class Generator(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz+10, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, 
                               stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), 100) 
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1).unsqueeze(2).unsqueeze(2)  # 合併輸入
        out = self.model(x)
        return out.view(x.size(0), 28, 28)