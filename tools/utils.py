"""Utils needed for model converter
Brought from avelab

make_dir and get_md5_hash implementations in avelab: https://github.rbx.com/Roblox/avelab/blob/24c084268756fadb95ed57a513c2512122795e35/avelab/common/utils/misc.py
write_json implementation in avelab: https://github.rbx.com/Roblox/avelab/blob/24c084268756fadb95ed57a513c2512122795e35/avelab/common/utils/json.py

"""
import os
import json
import hashlib


def make_dir(directory, exist_ok=True):
    """
    Savely create a new dir
    :param directory: str or pathlib.Path
        path to dir to be created
    :param exist_ok: bool
        avoids raising an error if dir already exists
    """
    directory = str(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    elif not exist_ok:
        raise Exception("directory {} already exists! ".format(directory))


def write_json(filename, dictionary, convert2str=False, **kwargs):
    """Writes a dictionary to a json

    Parameters
    ----------
    filename : str, pathlib.Path
        name of file to write
    dictionary : dict, OrderedDict
        dictionary to write to json
    convert2str : bool, optional
        convert all values to strings
    kwargs :
        Any remaining keyword arguments are passed to json.dump(...)

    """
    if convert2str:
        for k, v in dictionary.items():
            dictionary[k] = str(v)

    with open(str(filename), "w") as f:
        json.dump(dictionary, f, indent=2, **kwargs)


def get_md5_hash(file_paths):
    """get md5 hash of file or list of files

    :param file_paths: str or list of string to paths
    :return: str: md5 path
    """
    if not isinstance(file_paths, list):
        assert isinstance(file_paths, str)
        file_paths = [file_paths]

    hash_md5 = hashlib.md5()
    file_paths.sort()
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            hash_md5.update(f.read())
    return str(hash_md5.hexdigest())
