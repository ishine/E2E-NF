import subprocess
from logging import getLogger
from os import PathLike
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from formant_hifi_gan.utils import file_io

# TODO: logging in multiprocessing
logger = getLogger(__name__)


def path_create(file_list: list[PathLike], inputpath: PathLike, outputpath: PathLike, ext: str = None):
    for filepath in file_list:
        path_replace(filepath, inputpath, outputpath, ext)


def path_replace(filepath: PathLike, inputpath: PathLike, outputpath: PathLike, ext: str = None) -> Path:
    filepath = str.replace(str(filepath), str(inputpath), str(outputpath), 1)
    filepath = Path(filepath)
    if ext is not None:
        filepath = filepath.with_suffix(ext)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    return filepath


def aug_list_create(list_file: PathLike, path_list: list[PathLike]):
    """Create list of augment list

    Args:
        file_lists (str): Filename of wav list
        config (dict): Config
    """
    pth = Path(list_file)
    aug_list_file = pth.parent.joinpath("augment.scp")
    with open(aug_list_file, "w") as f:
        f.writelines(path_list)


def process(filepath: PathLike, config: DictConfig):
    is_aug = False
    sox_cmd = ["sox"]
    save_path = path_replace(
        filepath,
        config.in_dir,
        config.out_dir,
        ext=config.feature_format,
    )
    if np.random.rand() < config.rate_augmentation:
        read_sr = np.random.choice(
            np.arange(config.aug_sr_min, config.aug_sr_max + config.aug_sr_step, config.aug_sr_step)
        )
        sox_cmd += [
            "--rate",
            f"{read_sr}",
            f"{filepath}",
            "--rate",
            f"{config.sample_rate}",
            f"{save_path}",
            "tempo",
            f"{config.sample_rate/read_sr}",
        ]
        is_aug = True
    if config.norm_gain:
        norm_db = np.random.choice(
            np.arange(config.norm_db_min, config.norm_db_max + config.norm_db_step, config.norm_db_step)
        )
        sox_cmd += ["gain", "-n", f"{norm_db}"]
    if not (is_aug or config.norm_gain or config.save_no_aug):
        return None
    subprocess.run(sox_cmd)

    return str(save_path) + "\n"


@hydra.main(version_base=None, config_path="config", config_name="data_augment")
def main(config: DictConfig):
    # show default argument
    np.random.seed(config.seed)
    logger.info(OmegaConf.to_yaml(config))

    # read file list
    file_list = file_io.read_txt(to_absolute_path(config.file_list))
    logger.info(f"number of atterances = {len(file_list)}")
    file_and_conf_list = [(file, config) for file in file_list]
    path_create(file_list, config.in_dir, config.out_dir)

    augment_list = [
        r
        for r in tqdm(
            Parallel(n_jobs=config.n_jobs, return_as="generator")(
                (delayed(process)(path, config) for path, config in file_and_conf_list)
            ),
            total=len(file_list),
        )
        if r is not None
    ]
    aug_list_create(config.file_list, augment_list)


if __name__ == "__main__":
    main()
