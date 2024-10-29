import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 超參數
latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# 创建生成器实例
generator = Generator()

# 加载训练好的模型参数
generator.load_state_dict(torch.load('generator.pth',weights_only=False))
generator.eval()  # 设置为评估模式

# 生成潜在向量
batch_size = 16  # 你想生成的图片数量
latent_vectors = torch.randn(batch_size, latent_dim)

# 生成图片
with torch.no_grad():  # 不需要计算梯度
    generated_images = generator(latent_vectors)

# 将生成的图片转换到适合可视化的格式
generated_images = generated_images.view(batch_size, 1, 28, 28)  # 假设图片是 28x28
generated_images = (generated_images + 1) / 2  # 将 [-1, 1] 转换到 [0, 1]

# 可视化生成的图片
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(generated_images[i][0], cmap='gray')
    ax.axis('off')
plt.show()
