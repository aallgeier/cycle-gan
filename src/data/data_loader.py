from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import img_size


def get_data_loaders(data_pathA, data_pathB, batch_size, train="True"):

    if train:
        # Data Loaders
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    dataset_A = datasets.ImageFolder(data_pathA, transform=transform)
    dataset_B = datasets.ImageFolder(data_pathB, transform=transform)

    loader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
    loader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)

    return loader_A, loader_B