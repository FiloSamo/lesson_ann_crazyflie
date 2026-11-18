from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import transforms
from torch.onnx import export as onnx_export
import os

def load_datasets(data_path):
    '''
    Load the MNIST dataset and apply necessary transformations.
    Returns:
        tuple[Dataset, Dataset]: A tuple containing the training and testing datasets.
    '''
    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    return mnist_train, mnist_test


def save_onnx_format(model, dummy_input, onnx_path):
    """
    Save the PyTorch model in ONNX format for compatibility with other frameworks.
    """
    model.eval()
    onnx_export(model, dummy_input, onnx_path, opset_version=18)
    print(f"Model saved in ONNX format at: {onnx_path}")