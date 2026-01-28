"""Dataset wrapper that reads a CSV and converts each row into a HyperData sample.

Important behavior:
- Uses PyG InMemoryDataset caching: processed_file_names() controls the cache key.
  If you change feature extraction settings (mode/k/SMARTS), update the tag to avoid
  accidentally reusing stale processed .pt files.

"""

# data.py — dataset wrapper (now supports choosing hypergraph feature mode: FG or PG)
import os
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset

from utils import sequences_geodata_1
from device import device_info

class GeoDatasetBase_1(InMemoryDataset):
    """CSV -> HyperData dataset.

    index_x: list of CSV column indices used as input (concatenated as a string).
    index_y: CSV column index used as label (0/1).
    """
    def __init__(self, root, raw_name, index_x, index_y=None, has_targets=True,
                 hyper_mode: str = "pg",
                 pg_k: int = 3,
                 pad_dim_fg: int = None,
                 transform=None, pre_transform=None, **kwargs):
        self.filename = raw_name
        # Load the raw CSV once; process() will convert rows to HyperData objects.
        self.df = pd.read_csv(self.filename)
        self.index_x = index_x if isinstance(index_x, (list, tuple)) else [index_x]
        self.index_y = index_y
        self.has_targets = bool(has_targets and (index_y is not None))

        # Feature extraction / hypergraph configuration:
        self.hyper_mode = str(hyper_mode).lower().strip()
        self.pg_k = int(pg_k)
        self.pad_dim_fg = pad_dim_fg  # None => auto

        # PyG will call process() if processed file does not exist.
        super().__init__(root, transform, pre_transform)
        os.makedirs(self.processed_dir, exist_ok=True)
        # Load cached processed data (fast startup for repeated runs).
        data, slices = torch.load(self.processed_paths[0])
        self.data, self.slices = data, slices

    @property
    def raw_file_names(self):
        return [os.path.basename(self.filename)]

    @property
    def processed_file_names(self):
        base = os.path.splitext(os.path.basename(self.filename))[0]
        if self.hyper_mode == "pg":
            tag = f"pg_k{self.pg_k}_nolimit"
        elif self.hyper_mode == "fg":
            # If you version your SMARTS set, encode it here to avoid cache collisions.
            tag = "fg_smarts"
        else:
            tag = f"unknown_{self.hyper_mode}"
        return [f"{base}_hyper_{tag}.pt"]

    def download(self):
        pass

    def process(self):
        data_list = []
        cc = 0
        device = device_info().device  # used only by caller; tensors remain on CPU in preprocessing

        x_df = self.df.iloc[:, self.index_x].astype(str)
        # If multiple input columns are provided, we concatenate them.
        if x_df.shape[1] == 1:
            x_list = x_df.iloc[:, 0].tolist()
        else:
            # Row-wise concat: e.g., TCRalpha + TCRbeta + peptide -> one string.
            x_list = (x_df.apply(lambda r: ''.join(list(r.values)), axis=1)).tolist()

        if self.has_targets:
            y_list = self.df.iloc[:, self.index_y].astype(float).tolist()
        else:
            y_list = [0.0] * len(x_list)

        for x, y in zip(x_list, y_list):
            # Convert each sample into a HyperData object expected by Hyperpep.

            dp = sequences_geodata_1(
                cc=cc,
                sequence=x,
                y=y,
                device=device,
                mode=self.hyper_mode,
                k=self.pg_k,
                pad_dim_fg=self.pad_dim_fg
            )
            dp.x = dp.x_h  # used only for generating batch vector
            data_list.append(dp)
            cc += 1

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GeoDataset_1(GeoDatasetBase_1):
    def __init__(self, root, raw_name, index_x, index_y=None, has_targets=True, **kwargs):
        super().__init__(root, raw_name, index_x, index_y, has_targets, **kwargs)
