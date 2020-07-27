import torchvision

if __name__ == '__main__':
    torchvision.datasets.CIFAR10('./data', train=True, download=True)
    torchvision.datasets.CIFAR10('./data', train=False, download=True)