#     ______     ______     ______     __         ______     ______    
#    /\  ___\   /\  ___\   /\  ___\   /\ \       /\  __ \   /\  == \   
#    \ \  __\   \ \ \____  \ \___  \  \ \ \____  \ \  __ \  \ \  __<   
#     \ \_____\  \ \_____\  \/\_____\  \ \_____\  \ \_\ \_\  \ \_____\ 
#      \/_____/   \/_____/   \/_____/   \/_____/   \/_/\/_/   \/_____/ 
#                                                                      
# Author: Filippo Samorì
# Date: 2026-06-18
# Description: Train a simple network for the lesson.
#    


import os

import numpy as np
import torch
from torch import nn

from utils.models import NeuralNetwork
from utils.train import train
from utils.utils import save_onnx_format

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
PATH_OUT = os.path.join(os.path.dirname(__file__), "output")

# Network Architecture
NUM_INPUTS = 28 * 28
NUM_HIDDEN = 10
NUM_OUTPUTS = 10

def get_or_create_model(path, model, data_path, overwrite = False):
    if not os.path.exists(path) or overwrite:
        model = train(data_path=data_path, network_model=model)
        torch.save(model.state_dict(), path)
        return model
    model.load_state_dict(torch.load(path), strict=False)
    return model


def run():
    """Train, save, and generate other assets used for testing C implementation."""
    os.makedirs(PATH_OUT, exist_ok=True)
    model = NeuralNetwork(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
    model = get_or_create_model(os.path.join(PATH_OUT, "model.pk"), model=model, data_path=DATA_PATH, overwrite=False)
    model.eval()
    dummy_input = torch.randn(1, 28 * 28)
    save_onnx_format(model, dummy_input, os.path.join(PATH_OUT, "model.onnx"))


if __name__ == '__main__':
    run()
