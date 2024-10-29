import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import time

# 超參數
latent_dim = 100  # 噪聲向量的維度
batch_size = 64
lr = 0.0002
epochs = 50

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh()
            
        )

    def forward(self, x):
        x = self.model(x)
        # print(x.shape)
        return x

# 判別器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 初始化模型
G = Generator()
D = Discriminator()
G = Generator().to(device)
D = Discriminator().to(device)

# 損失函數和優化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

# 資料集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
mnist = datasets.MNIST(root='data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# 訓練過程
for epoch in range(epochs):
    start = time.time()
    for i, (real_imgs, _) in enumerate(dataloader):
        # 準備真實和偽造標籤
        real = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # 訓練判別器
        optimizer_D.zero_grad()
        real_imgs = real_imgs.view(batch_size, -1).to(device)
        if (real_imgs.shape) == torch.Size([64, 392]):
            continue
        real_loss = criterion(D(real_imgs), real)
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(noise)
        fake_loss = criterion(D(fake_imgs.detach()), fake)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # 訓練生成器
        optimizer_G.zero_grad()
        g_loss = criterion(D(fake_imgs), real)
        g_loss.backward()
        optimizer_G.step()
    
    
    print(f"Epoch [{epoch}/{epochs}]  Loss D: {d_loss:.4f}, loss G: {g_loss:.4f}, time: {time.time()-start}")

# 儲存生成器的權重
torch.save(G.state_dict(), "generator.pth")
