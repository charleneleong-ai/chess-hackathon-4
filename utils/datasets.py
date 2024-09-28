import os
import numpy as np
import torch
from torch.utils.data import Dataset
from itertools import accumulate
from h5py import File as h5pyFile
import h5py
from pathlib import Path

class PGN_HDF_Dataset(Dataset):
    def __init__(self, source_dir=None, meta=False):
        self.source_dir = Path(source_dir) if isinstance(source_dir, str) else source_dir
        self.meta = meta
        with open(os.path.join(self.source_dir, "inventory.txt"), "r") as file:
            self.inventory = file.readlines()
        sizes, self.filenames = zip(*[i.split() for i in self.inventory[1:]])
        self.sizes = [int(s) for s in sizes]
        self.len = sum(self.sizes)
        self.breaks = np.array(list(accumulate(self.sizes)))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        hdf_idx = (self.breaks > idx).argmax().item()
        pgn_idx = idx - sum(self.sizes[:hdf_idx])
        hdf_path = os.path.join(self.source_dir, self.filenames[hdf_idx])
        with h5pyFile(hdf_path, 'r') as hf:
            pgn = hf["pgn"][pgn_idx].decode('utf-8')
            if self.meta:
                meta = hf["meta"][pgn_idx].decode('utf-8')
        if self.meta:
            return pgn, meta
        else:
            return pgn
    
class EVAL_HDF_Dataset(Dataset):
    def __init__(self, source_dir):
        super().__init__()
        self.source_dir = Path(source_dir) if isinstance(source_dir, str) else source_dir
        # with open(os.path.join(self.source_dir, "inventory.txt"), "r") as file:
        #     self.inventory = file.readlines()
        # sizes, self.filenames = zip(*[i.split() for i in self.inventory[1:]])
        # self.sizes = [int(s) for s in sizes]
        # self.len = sum(self.sizes)
        # self.breaks = np.array(list(accumulate(self.sizes)))
        
        self.filenames = sorted(self.source_dir.glob("evalHDF*"))
        
        # Read the sizes of datasets from the files
        self.sizes = []
        for filename in self.filenames:
            try:
                with h5pyFile(filename, 'r') as hf:
                    size = len(hf["boards"])
                    self.sizes.append(size)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

        # Dynamically calculate the total length and index breaks
        self.len = sum(self.sizes)
        self.breaks = np.array(list(accumulate(self.sizes)))


    def __len__(self):
        return self.len
            
    def __getitem__(self, idx):
        hdf_idx = (self.breaks > idx).argmax().item()
        board_idx = idx - sum(self.sizes[:hdf_idx])
        hdf_path = os.path.join(self.source_dir, self.filenames[hdf_idx])
        
        try:
            with h5pyFile(hdf_path, 'r') as hf:
                board = hf["boards"][board_idx]
                score = hf["scores"][board_idx]
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {hdf_path}: {e}")

        board = torch.from_numpy(board)
        score = torch.tensor(score)     
        return board, score