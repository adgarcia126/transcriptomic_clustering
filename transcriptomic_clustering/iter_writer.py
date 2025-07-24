import logging

import numpy as np
import scipy as scp
import anndata as ad
import h5py
from anndata.io import write_elem

logger = logging.getLogger(__name__)

class AnnDataIterWriter():
    """
    Class to handle iteratively writing filebacked AnnData Objects
    """
    def __init__(self, filename, initial_chunk, obs, var, obsm, dtype=None):
        self.issparse = scp.sparse.issparse(initial_chunk)
        self.initialize_file(filename, initial_chunk, obs, var, obsm, dtype=dtype)
        self.adata = ad.read_h5ad(filename, backed='r+')


    def initialize_file(self, filename, initial_chunk, obs, var, obsm, dtype=None):
        with h5py.File(filename, "w") as f:
            if self.issparse:
                if dtype is not None:
                    initial_chunk = initial_chunk.astype(dtype)
                else:
                    initial_chunk = initial_chunk.astype(np.float32)
            else:
                if dtype is not None:
                    initial_chunk = np.asarray(initial_chunk, dtype=dtype)
                else:
                    initial_chunk = np.asarray(initial_chunk, dtype=np.float32)
    
            initial_chunk = np.atleast_2d(initial_chunk)
            write_elem(f, "X", initial_chunk)
            write_elem(f, "obs", obs)
            write_elem(f, "var", var)
    
            f.create_group("obsm")
            for key, val in obsm.items():
                write_elem(f, f"obsm/{key}", val)
            f["obsm"].attrs["__keys__"] = list(obsm.keys())
            write_elem(f, "obs", obs)
            write_elem(f, "var", var)
            for key, val in obsm.items():
                write_elem(f, f"obsm/{key}", val)  # Store each key separately


    def add_chunk(self, chunk):
        if self.issparse:
            self.adata.X.append(chunk)
        else:
            chunk = np.atleast_2d(chunk)
            chunk_nrows = chunk.shape[0]
            self.adata.X.resize(
                (self.adata.X.shape[0] + chunk_nrows),
                axis = 0
            )
            self.adata.X[-chunk_nrows:] = chunk
