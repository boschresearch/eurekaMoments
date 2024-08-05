import os
import numpy as np
import torch
import torchvision


def download_mnist(root_dir, out_dir):
    mnist_trainset = torchvision.datasets.MNIST(root=root_dir, train=True, download=True)
    mnist_testset = torchvision.datasets.MNIST(root=root_dir, train=False, download=True)
    train_data = (
        mnist_trainset.data,
        mnist_trainset.targets
    )
    test_data = (
        mnist_testset.data,
        mnist_testset.targets
    )
    torch.save(train_data, os.path.join(out_dir, 'training.pt'))
    torch.save(test_data, os.path.join(out_dir, 'test.pt'))


def download_fashion_mnist(root_dir, out_dir):
    fashion_mnist_trainset = torchvision.datasets.FashionMNIST(root=root_dir, train=True, download=True)
    fashion_mnist_testset = torchvision.datasets.FashionMNIST(root=root_dir, train=False, download=True)
    fashion_data = {
        'x_train': fashion_mnist_trainset.data,
        'y_train': fashion_mnist_trainset.targets,
        'x_test': fashion_mnist_testset.data,
        'y_test': fashion_mnist_testset.targets
    }
    np.savez_compressed(os.path.join(out_dir, 'fashion_mnist.npz'), **fashion_data)


if __name__ == '__main__':
    root_dir = './data'
    processed_dir = os.path.join(root_dir, 'mnist/processed')
    fashion_dir = os.path.join(root_dir, 'fashion_mnist')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(fashion_dir, exist_ok=True)

    download_mnist(root_dir, processed_dir)
    download_fashion_mnist(root_dir, fashion_dir)
