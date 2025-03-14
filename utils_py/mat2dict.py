import scipy.io
import numpy as np

def loadmat_to_dict(filename):
    """
    Load a MATLAB .mat file and convert MATLAB structures into nested Python dictionaries.
    Additionally, for arrays of characters, join them into a single string per row and strip
    leading/trailing whitespace.

    Parameters:
        filename (str): The path to the MATLAB file (.mat) to be loaded.

    Returns:
        dict: A dictionary representation of the MATLAB file contents, with MATLAB-specific 
              structures converted to Python dictionaries and character arrays converted to strings.

    Example:
        >>> data_dict = loadmat_to_dict('data.mat')
        >>> print(data_dict.keys())
    """
    def _check_keys(d):
        # Recursively process each key-value pair in the dictionary.
        for key in d:
            d[key] = _process_item(d[key])
        return d

    def _process_item(item):
        # Convert MATLAB structs, dictionaries, NumPy arrays, and lists appropriately.
        if isinstance(item, scipy.io.matlab.mat_struct):
            return _todict(item)
        elif isinstance(item, dict):
            return _check_keys(item)
        elif isinstance(item, np.ndarray):
            return _process_array(item)
        elif isinstance(item, list):
            return [ _process_item(i) for i in item ]
        else:
            return item

    def _todict(matobj):
        # Recursively convert a mat_struct object to a dictionary.
        d = {}
        for field in matobj._fieldnames:
            elem = getattr(matobj, field)
            d[field] = _process_item(elem)
        return d

    def _process_array(arr):
        # If the array is of a string type (each element being a single character),
        # join them into a string (per row if 2D) and strip whitespace.
        if arr.dtype.kind in ['U', 'S']:
            if arr.ndim == 2:
                # Assume each row represents a string.
                return [''.join(row).strip() for row in arr]
            elif arr.ndim == 1:
                return ''.join(arr).strip()
            else:
                # For arrays with more dimensions, flatten to a string.
                return ''.join(arr.flatten()).strip()
        # For other array types, process each element recursively.
        return np.array([_process_item(x) for x in arr])

    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)
