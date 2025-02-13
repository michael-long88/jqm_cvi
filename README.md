jqmcvi
======

Small module with Cluster Validity Indices (CVI)
------------------------------------------------

Dunn and Davius Bouldin indices are implemented. It follows the equations presented in theory.pdf.

> base.py : Python + NumPy
>
> basec.pyx : Python + NumPy optimized with Cython
>
> basec.pyx tested on M1 Macbook Pro, Python 3.9 and compiled with `python setup.py build_ext -i`

Functions:
----------

**dunn(k_list)**:

> Slow implementation of Dunn index that depends on numpy
>
> -- basec.pyx Cython implementation is much faster but slower than dunn_fast()

```python
	""" Dunn index [CVI]
  
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
```

**dunn_fast(points, labels)**:

> Fast implementation of Dunn index that depends on numpy and sklearn.pairwise
>
> -- No Cython implementation

```python
	""" Dunn index - FAST (using sklearn pairwise euclidean_distance function)
  
    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
```

**davisbouldin(k_list, k_centers)**:

> Implementation of Davis Boulding index that depends on numpy
>
> -- basec.pyx Cython implementation is much faster

```python
	""" Davis Bouldin Index

	Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    k_centers : np.array
        The array of the cluster centers (prototypes) of type np.array([K, p])
    """
```

## Developing

- Install dependencies
  - `pip install -r requirements.txt`
- Recompile `basec.c` based on updated dependencies, updated Python version, or changes to `basec.pyx`
  - `python setup.py build_ext -i`
- (Re)Install `jqmcvi` after any changes
  - `pip install -e .`

## Testing

If running from the console, make sure present working directory is `tests/` before running `python cvi-dev.py`.
