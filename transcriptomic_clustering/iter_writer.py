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
            if dtype is not None and not self.issparse:
                initial_chunk = np.asarray(initial_chunk, dtype=dtype)
    
            initial_chunk = np.atleast_2d(initial_chunk)
            write_elem(f, "X", initial_chunk)
            write_elem(f, "obs", obs)
            write_elem(f, "var", var)
    
            f.create_group("obsm")
            for key, val in obsm.items():
                write_elem(f, f"obsm/{key}", val)
            else:
                if dtype is None:
                    dtype = initial_chunk.dtype
                initial_chunk = np.atleast_2d(initial_chunk)
                ad._io.h5ad.write_array(
                    f, "X", initial_chunk,
                    dataset_kwargs={
                        'maxshape': (None, initial_chunk.shape[1]),
                        'dtype': dtype
                    }
                )
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
