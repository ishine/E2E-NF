import csv
import glob
import os
import sys
from logging import getLogger
from os import PathLike

import h5py
import numpy as np

# A logger for this file
logger = getLogger(__name__)


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logger.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logger.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                hdf5_file.__delitem__(hdf5_path)
            else:
                logger.error(
                    "Dataset in hdf5 file already exists. if you want to overwrite, please set is_overwrite = True."
                )
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


def check_hdf5(hdf5_name, hdf5_path):
    """Check hdf5 file existence

    Args:
        hdf5_name (str): filename of hdf5 file
        hdf5_path (str): dataset name in hdf5 file

    Return:
        (bool): dataset exists then return true

    """
    if not os.path.exists(hdf5_name):
        return False
    else:
        with h5py.File(hdf5_name, "r") as f:
            if hdf5_path in f:
                return True
            else:
                return False


def read_txt(file_list):
    """Read .txt file list

    Arg:
        file_list (str): txt file filename

    Return:
        (list): list of read lines

    """
    with open(file_list, "r") as f:
        filenames = f.readlines()
    return [filename.replace("\n", "") for filename in filenames]


def write_txt(path: PathLike, dict_text: dict[str]):
    with open(path, "w", encoding="utf-8") as f:
        for key, value in dict_text.items():
            f.write(f"{key}:\n{value}\n")


def write_csv(path: PathLike, list_text: list):
    with open(path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(list_text)


def check_filename(list1, list2):
    """Check the filenames of two list are matched

    Arg:
        list1 (list): file list 1
        list2 (list): file list 2

    Return:
        (bool): matched (True) or not (False)

    """

    def _filename(x):
        return os.path.basename(x).split(".")[0]

    list1 = list(map(_filename, list1))
    list2 = list(map(_filename, list2))

    return list1 == list2


def find_files(path: str, pattern: str):
    """Find files in directory.
    Args:
        directory (str): Directory to find files.
        pattern (str, optional): Pattern to find files. Defaults to "**/*.wav".
    Returns:
        list: List of files.
    """

    files = glob.glob(f"{path}/{pattern}", recursive=True)
    files.sort()

    return files
