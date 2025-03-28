import logging
import numpy as np
import scipy.sparse as sp
import anndata as ad
import h5py

logger = logging.getLogger(__name__)

class AnnDataIterWriter:
    """
    Class to handle iteratively writing file-backed AnnData objects
    """
    def __init__(self, filename, initial_chunk, obs, var, dtype=None):
        self.issparse = sp.issparse(initial_chunk)
        self.initialize_file(filename, initial_chunk, obs, var, dtype=dtype)
        self.adata = ad.read_h5ad(filename, backed='r+')

    def initialize_file(self, filename, initial_chunk, obs, var, dtype=None):
        """Initializes the H5AD file using the initial chunk"""
        
        with h5py.File(filename, "w") as f:
            if self.issparse:
                if dtype is not None:
                    logger.warning("Ignoring dtype for sparse matrix")
                
                # Convert sparse matrix indices to int64 for compatibility
                initial_chunk = initial_chunk.tocsr()
                initial_chunk.indptr = initial_chunk.indptr.astype(np.int64)
                initial_chunk.indices = initial_chunk.indices.astype(np.int64)
                
                # Store sparse matrix as a group
                grp = f.create_group("X")
                grp.create_dataset("data", data=initial_chunk.data)
                grp.create_dataset("indices", data=initial_chunk.indices)
                grp.create_dataset("indptr", data=initial_chunk.indptr)
                grp.attrs["shape"] = initial_chunk.shape
            else:
                if dtype is None:
                    dtype = initial_chunk.dtype
                initial_chunk = np.atleast_2d(initial_chunk)
                f.create_dataset("X", data=initial_chunk, maxshape=(None, initial_chunk.shape[1]), dtype=dtype)

            ad._io.h5.write_dataframe(f, "obs", obs)
            ad._io.h5.write_dataframe(f, "var", var)

    def add_chunk(self, chunk):
        """Appends a chunk to the existing dataset"""
        if self.issparse:
            existing = sp.csr_matrix((self.adata.X.data, self.adata.X.indices, self.adata.X.indptr), shape=self.adata.X.shape)
            new_matrix = sp.vstack([existing, chunk])
            
            with h5py.File(self.adata.filename, "r+") as f:
                grp = f["X"]
                grp["data"][...] = new_matrix.data
                grp["indices"][...] = new_matrix.indices
                grp["indptr"][...] = new_matrix.indptr
                grp.attrs["shape"] = new_matrix.shape
        else:
            chunk = np.atleast_2d(chunk)
            chunk_nrows = chunk.shape[0]
            with h5py.File(self.adata.filename, "r+") as f:
                dset = f["X"]
                dset.resize((dset.shape[0] + chunk_nrows), axis=0)
                dset[-chunk_nrows:] = chunk
