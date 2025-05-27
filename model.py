import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional

class FFN(nn.Module):
    def __init__(self,input_dim:int,hidden_dim:int,output_dim:int):
        super().__init__()
        self.input2hidden = nn.Linear(input_dim,hidden_dim)
        self.hidden2hidden = nn.Linear(hidden_dim,hidden_dim)
        self.hidden2output = nn.Linear(hidden_dim,output_dim)
    
    def forward(self,input:Tensor):
        """Compute a forward pass"""

        first_layer = torch.tanh(self.input2hidden(input))
        second_layer = torch.tanh(self.hidden2hidden(first_layer))
        output_layer = self.hidden2output(second_layer)
        return output_layer  #torch.softmax(output_layer)
    

    # This class should return [output_dim] # of logits. It is our job to then softmax them during training
    # (AKA don't put softmax inside the forward function of this class)

    #Also use torch.save() and torch.load() to save model weights to a ".pt" file to be loaded later.
    # default behavior of torch.load() is weight_only=True. Switch to false for full model.