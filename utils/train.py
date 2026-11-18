import os
import ssl

import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from .utils import load_datasets

# Necessary to download MNIST dataset.
ssl._create_default_https_context = ssl._create_unverified_context

def train(data_path, network_model, num_epochs = 1):
    # Load the network onto CUDA if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = network_model.to(device)

    # Create DataLoaders
    batch_size = 128
    mnist_train, mnist_test = load_datasets(data_path)
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

    dtype = torch.float

    loss_hist = []
    test_loss_hist = []
    counter = 0

    # Outer training loop
    for i in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)
        print(f"Epoch {i+1}/{num_epochs}")

        # Minibatch training loop
        for data, target in tqdm(train_batch):
            data = data.to(device)
            target = target.to(device)

            # forward pass
            model.train()
            out = model(data.view(batch_size, -1)) 
            
            # initialize the loss & sum over time
            loss_val = loss(out, target)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Test set
            with torch.no_grad():

                test_data, test_target = next(iter(test_loader))
                test_data = test_data.to(device)
                test_target = test_target.to(device)

                test_out= model(test_data.view(batch_size, -1))
                    
                # Test set loss
                test_loss = loss(test_out, test_target)

                counter += 1
                iter_counter +=1

    return model.to(torch.device("cpu"))

