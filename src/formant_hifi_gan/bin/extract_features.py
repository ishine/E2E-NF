from logging import getLogger
from os import PathLike
from pathlib import Path

import hydra
import numpy as np
import torch
from formant_hifi_gan.features import f0_utils, fixed, praat, spectral, spectrogram
from formant_hifi_gan.utils import audio_io, file_io, filter
from hydra.utils import to_absolute_path
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# TODO: logging in multiprocessing
logger = getLogger(__name__)


def path_create(file_list: list[PathLike], outputpath: PathLike, ext: str = None):
    for filepath in file_list:
        path_replace(filepath, outputpath, ext)


def path_replace(filepath: PathLike, outputpath: PathLike, ext: str = None) -> Path:
    filepath, outputpath = Path(filepath), Path(outputpath)
    if ext is not None:
        filepath = filepath.with_suffix(ext)
    outputpath.mkdir(parents=True, exist_ok=True)
    filepath = outputpath.joinpath(filepath.name)

    return filepath


def aux_list_create(file_list: list[PathLike], config: DictConfig):
    """Create list of auxiliary acoustic features

    Args:
        wav_list_file (str): Filename of wav list
        config (dict): Config

    """
    aux_list_file = file_list.replace(".scp", ".list")
    wav_files = file_io.read_txt(file_list)
    with open(aux_list_file, "w") as f:
        for wav_name in wav_files:
            feat_name = path_replace(
                wav_name,
                config.out_dir,
                ext=config.feature_format,
            )
            f.write(f"{feat_name}\n")


def process(filepath: PathLike, config: DictConfig):
    logger = getLogger(__name__)
    wav, sr = audio_io.load_wav(filepath, target_sr=config.sample_rate)
    if (wav == 0).all():
        logger.warning(f"missing file: {filepath}")

        return None
    if config.highpass_cutoff > 0:
        wav = filter.low_cut_filter(wav, sr, config.highpass_cutoff)
    to_stft = spectrogram.STFT(
        sr=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        win_size=config.win_size,
        hop_length=config.hop_length,
        fmin=config.fmin,
        fmax=config.fmax,
        clip_val=config.clip_val,
    )
    f0_extractor = f0_utils.F0_Extractor(
        "harvest",
        sample_rate=config.sample_rate,
        hop_size=config.hop_length,
        f0_min=config.minf0,
        f0_max=config.maxf0,
        fix_by_reaper=False,
    )

    spec = to_stft.get_linear(torch.from_numpy(wav.astype(np.float32)))
    mfbsp = to_stft.to_mel(spec, compress=True)
    mfbsp = mfbsp.numpy()

    f0 = f0_extractor.extract(wav, return_time=False)
    uv, cf0, is_all_uv = fixed.to_continuous(f0)
    if is_all_uv:
        lpf_fs = int(config.sample_rate / config.hop_length)
        cf0_lpf = filter.low_pass_filter(cf0, lpf_fs, cutoff=20)
        next_cutoff = 70
        while not (cf0_lpf >= [0]).all():
            cf0_lpf = filter.low_pass_filter(cf0, lpf_fs, cutoff=next_cutoff)
            next_cutoff *= 2
    else:
        logger.warning(f"all frame is unvoiced: {filepath}")
        return None

    formants, fo_time = praat.get_optimized_formants(
        wav,
        sr=config.sample_rate,
        hop_length=config.hop_length,
        win_size=config.fo_win_size,
        pre_enphasis=config.pre_enphasis,
        search_fo_max=config.search_fo_max,
        search_fo_min=config.search_fo_min,
        search_step=config.search_step,
    )
    uv, f0, cf0_lpf, spec, mfbsp, formants = fixed.adjust_min_len([uv, f0, cf0_lpf, spec, mfbsp, formants])
    assert uv.shape[0] == f0.shape[0] == cf0_lpf.shape[0] == spec.shape[-1] == mfbsp.shape[-1] == formants.shape[-1]

    f1, f2, f3, f4 = formants[0], formants[1], formants[2], formants[3]
    _, cf1, _ = fixed.to_continuous(f1)
    _, cf2, _ = fixed.to_continuous(f2)
    _, cf3, _ = fixed.to_continuous(f3)
    _, cf4, _ = fixed.to_continuous(f4)

    spec = spec.numpy()
    centroid = spectral.get_centroid(spec, sr=config.sample_rate, n_fft=config.n_fft)
    slope = spectral.get_slope(spec)
    energy = spectral.get_energy(spec)

    # expand to (T, 1)
    uv = np.expand_dims(uv, axis=-1)
    f0 = np.expand_dims(f0, axis=-1)
    f1 = np.expand_dims(f1, axis=-1)
    f2 = np.expand_dims(f2, axis=-1)
    f3 = np.expand_dims(f3, axis=-1)
    f4 = np.expand_dims(f4, axis=-1)
    cf0_lpf = np.expand_dims(cf0_lpf, axis=-1)
    cf1 = np.expand_dims(cf1, axis=-1)
    cf2 = np.expand_dims(cf2, axis=-1)
    cf3 = np.expand_dims(cf3, axis=-1)
    cf4 = np.expand_dims(cf4, axis=-1)
    slope = np.expand_dims(slope, axis=-1)
    energy = np.expand_dims(energy, axis=-1)

    feat_name = path_replace(
        filepath,
        config.out_dir,
        ext=config.feature_format,
    )
    feat_name = to_absolute_path(feat_name)

    # feature shape (T, 1) or (T, D)
    file_io.write_hdf5(feat_name, "/uv", uv)
    file_io.write_hdf5(feat_name, "/f0", f0)
    file_io.write_hdf5(feat_name, "/f1", f1)
    file_io.write_hdf5(feat_name, "/f2", f2)
    file_io.write_hdf5(feat_name, "/f3", f3)
    file_io.write_hdf5(feat_name, "/f4", f4)
    file_io.write_hdf5(feat_name, "/cf0", cf0_lpf)
    file_io.write_hdf5(feat_name, "/cf1", cf1)
    file_io.write_hdf5(feat_name, "/cf2", cf2)
    file_io.write_hdf5(feat_name, "/cf3", cf3)
    file_io.write_hdf5(feat_name, "/cf4", cf4)
    file_io.write_hdf5(feat_name, "/slope", slope)
    file_io.write_hdf5(feat_name, "/energy", energy)
    file_io.write_hdf5(feat_name, "/centroid", centroid.T)
    file_io.write_hdf5(feat_name, "/mfbsp", mfbsp.T)


@hydra.main(version_base=None, config_path="config", config_name="extract_features")
def main(config: DictConfig):
    # show default argument
    logger.info(OmegaConf.to_yaml(config))

    # read file list
    file_list = file_io.read_txt(to_absolute_path(config.file_list))
    logger.info(f"number of atterances = {len(file_list)}")

    configs = [config] * len(file_list)

    # set mode
    if config.inv:
        # create auxiliary feature list
        aux_list_create(to_absolute_path(config.file_list), config)
        # create folder
        path_create(file_list, config.out_dir, config.feature_format)

    _ = [
        r
        for r in tqdm(
            Parallel(n_jobs=config.n_jobs, return_as="generator")(
                (delayed(process)(path, config) for path, config in zip(file_list, configs))
            ),
            total=len(file_list),
        )
    ]


if __name__ == "__main__":
    main()
