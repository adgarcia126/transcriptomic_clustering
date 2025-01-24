import logging
import numpy as np
import scipy.sparse as sp
import anndata as ad
import h5py

logger = logging.getLogger(__name__)

class AnnDataIterWriter:
    """
    Class to handle iteratively writing file-backed AnnData Objects
    """

    def __init__(self, filename, initial_chunk, obs, var, dtype=None):
        self.issparse = sp.issparse(initial_chunk)
        self.initialize_file(filename, initial_chunk, obs, var, dtype=dtype)
        self.adata = ad.read_h5ad(filename, backed="r+")

    def initialize_file(self, filename, initial_chunk, obs, var, dtype=None):
        """Uses initial chunk to determine group type"""
        with h5py.File(filename, "w") as f:
            # Write the data matrix (X)
            if self.issparse:
                if dtype is not None:
                    logger.warning("Ignoring dtype for sparse matrix")
                # Ensure sparse matrix index types are compatible
                initial_chunk.indptr = initial_chunk.indptr.astype(np.int64)
                initial_chunk.indices = initial_chunk.indices.astype(np.int64)

                # Write sparse matrix data
                group = f.create_group("X")
                group.create_dataset("data", data=initial_chunk.data)
                group.create_dataset("indices", data=initial_chunk.indices)
                group.create_dataset("indptr", data=initial_chunk.indptr)
                group.attrs["shape"] = initial_chunk.shape
                group.attrs["format"] = "csr"
            else:
                if dtype is None:
                    dtype = initial_chunk.dtype
                initial_chunk = np.atleast_2d(initial_chunk)
                f.create_dataset(
                    "X",
                    data=initial_chunk,
                    maxshape=(None, initial_chunk.shape[1]),
                    dtype=dtype,
                )

            # Write observation (obs) and variable (var) data
            f.create_group("obs")
            for col in obs.columns:
                f["obs"].create_dataset(col, data=obs[col].values)
            f["obs"].attrs["columns"] = list(obs.columns)

            f.create_group("var")
            for col in var.columns:
                f["var"].create_dataset(col, data=var[col].values)
            f["var"].attrs["columns"] = list(var.columns)

    def add_chunk(self, chunk):
        """Add a chunk of data to the file-backed AnnData object"""
        if self.issparse:
            # Append to sparse matrix (not directly supported in h5py, requires manual handling)
            chunk = sp.csr_matrix(chunk)
            with h5py.File(self.adata.filename, "r+") as f:
                group = f["X"]
                group["data"].resize((group["data"].shape[0] + len(chunk.data)), axis=0)
                group["data"][-len(chunk.data) :] = chunk.data

                group["indices"].resize(
                    (group["indices"].shape[0] + len(chunk.indices)), axis=0
                )
                group["indices"][-len(chunk.indices) :] = chunk.indices

                group["indptr"].resize(
                    (group["indptr"].shape[0] + len(chunk.indptr) - 1), axis=0
                )
                group["indptr"][-(len(chunk.indptr) - 1) :] = chunk.indptr[1:]

                group.attrs["shape"] = (
                    group.attrs["shape"][0] + chunk.shape[0],
                    group.attrs["shape"][1],
                )
        else:
            # Append to dense matrix
            chunk = np.atleast_2d(chunk)
            chunk_nrows = chunk.shape[0]
            with h5py.File(self.adata.filename, "r+") as f:
                dataset = f["X"]
                dataset.resize((dataset.shape[0] + chunk_nrows), axis=0)
                dataset[-chunk_nrows:] = chunk
