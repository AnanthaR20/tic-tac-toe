import re,sys,random
from typing import Any
import numpy as np
import pandas as pd

# Type Alias
Example = dict[str, Any]

class Dataset:
    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    def example_to_tensors(self, index: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def batch_as_tensors(self, start: int, end: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]

    def __len__(self) -> int:
        return len(self.examples)
    

class TTTDataset(Dataset):

    labels_to_move = {0: (0,0), 1: (0,1), 2: (0,2), 3: (1,0), 4: (1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)}
    label_one_hots = np.eye(len(labels_to_move))

    def example_to_tensors(self, index:int) -> dict[str,np.ndarray]:
        """Convert a given example to Tensor format
        
        Args:
            index: desired example
        
        Returns:
            The same example, but in Tensor form and with one-hot vector for label
        """
        example = self.__getitem__(index)
        return {
            "input": np.array(example["input"]),
            "label": TTTDataset.label_one_hots[example["label"]]
        }
    
    def batch_as_tensors(self, start: int, end: int) -> dict[str, np.ndarray]:
        examples = [self.example_to_tensors(index) for index in range(start, end)]
        return {
            "input": np.stack([example["input"] for example in examples]),
            "label": np.stack([example["label"] for example in examples]),
        }

    def from_txt(file:str):
        """Return a Dataset from a text file.
        
        Args:
            file: filename
        """
        label_splitter = re.compile(r"[\d-]+")
        # feature_splitter = re.compile(r"[^,]")
        l = []
        
        with open(file, 'r') as f:
            file_as_list = f.read().splitlines()
            for i in file_as_list:
                label = int(label_splitter.findall(i)[0])
                features = label_splitter.findall(i)[1:]
                # print(label_splitter.findall(i))
                # print(features)
                features = [int(j) for j in features]

                l.append({
                    "input": features,
                    "label": label
                })
        
        return TTTDataset(examples=l)

