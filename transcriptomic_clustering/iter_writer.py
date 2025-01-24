import logging
import numpy as np
import scipy.sparse as sp
import anndata as ad

logger = logging.getLogger(__name__)


class AnnDataIterWriter:
    """
    Class to handle iteratively writing file-backed AnnData Objects
    """
    def __init__(self, filename, initial_chunk, obs, var, dtype=None):
        self.issparse = sp.issparse(initial_chunk)
        self.filename = filename
        self.initialize_file(filename, initial_chunk, obs, var, dtype=dtype)
        self.adata = ad.read_h5ad(filename, backed="r+")

    def initialize_file(self, filename, initial_chunk, obs, var, dtype=None):
        """Uses the initial chunk to initialize the AnnData object."""
        if self.issparse:
            if dtype is not None:
                logger.warning("Ignoring dtype for sparse matrix.")
            initial_chunk = sp.csr_matrix(initial_chunk)  # Ensure CSR format
            adata = ad.AnnData(X=initial_chunk, obs=obs, var=var)
        else:
            if dtype is None:
                dtype = initial_chunk.dtype
            initial_chunk = np.atleast_2d(initial_chunk).astype(dtype)
            adata = ad.AnnData(X=initial_chunk, obs=obs, var=var)

        # Save the initialized AnnData object in file-backed mode
        adata.write(filename)

    def add_chunk(self, chunk):
        """Adds a new chunk of data to the file-backed AnnData object."""
        if self.issparse:
            if not sp.issparse(chunk):
                raise ValueError("Expected a sparse chunk, but got dense data.")
            chunk = sp.csr_matrix(chunk)  # Ensure CSR format
            self.adata.file["X"].append(chunk)
        else:
            chunk = np.atleast_2d(chunk)
            chunk_nrows = chunk.shape[0]

            # Resize and add new rows
            current_nrows = self.adata.shape[0]
            self.adata.file["X"].resize((current_nrows + chunk_nrows, self.adata.shape[1]))
            self.adata.file["X"][-chunk_nrows:] = chunk
