import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import SimpleNet
import numpy as np
import random

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def main():
    # 设置随机种子确保可重现性
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 设置确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    # 设置确定性的数据加载器
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64, shuffle=True, generator=generator)

    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 3): # Train for 2 epochs for demonstration
        train(model, device, train_loader, optimizer, epoch)

    torch.save(model.state_dict(), "mnist_simple.pt")

if __name__ == '__main__':
    main()
