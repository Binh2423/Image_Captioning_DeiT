import sys
import os
sys.path.append('../..')
from definitions import *

class DatasetDescriptor:

    LOOKUP = {
        "flickr30k" : "D:/DACN/dataset/flickr30k"
    }

    def __init__(self, root : str) -> None:
        
        self.root = root

        self.images_dir = os.path.join(root, 'images')
        self.train_captions = os.path.join(root, 'train_captions.csv')
        self.test_captions = os.path.join(root, 'test_captions.csv')
        self.captions = os.path.join(root, 'captions.txt')
        self.vocab_path = os.path.join(root, "vocab.pkl")

    def get_by_name(name : str):

        root = DatasetDescriptor.LOOKUP.get(name)
        
        if root is None:
            raise Exception(f"{name} is not a valid dataset name")

        return DatasetDescriptor(root)