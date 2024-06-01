from logging import getLogger
from multiprocessing import Manager
from os import PathLike
from typing import Optional

import numpy as np
from hydra.utils import to_absolute_path
from joblib import load
from torch.utils.data import Dataset

from formant_hifi_gan.features.fixed import get_segment, validate_length
from formant_hifi_gan.utils.audio_io import load_wav
from formant_hifi_gan.utils.file_io import check_filename, read_hdf5, read_txt

# A logger for this file
logger = getLogger(__name__)


class AudioFeatDataset(Dataset):
    """PyTorch compatible audio and acoustic feat. dataset."""

    def __init__(
        self,
        stats: PathLike,
        audio_list: PathLike,
        feat_list: PathLike,
        audio_length_threshold: Optional[int] = None,
        feat_length_threshold: Optional[int] = None,
        return_filename: bool = False,
        allow_cache: bool = False,
        sample_rate: int = 24000,
        hop_size: int = 120,
        aux_feats: list[str] = [
            "uv",
            "lcf0",
            "cf1",
            "cf2",
            "cf3",
            "cf4",
            "slope",
            "centroid",
            "energy",
        ],
        f0_factor: float = 1.0,
        formants_factor: list[float] = [1.0, 1.0, 1.0, 1.0],
    ):
        """Initialize dataset.

        Args:
            stats (str): Filename of the statistic hdf5 file.
            audio_list (str): Filename of the list of audio files.
            feat_list (str): Filename of the list of feature files.
            audio_length_threshold (int): Threshold to remove short audio files.
            feat_length_threshold (int): Threshold to remove short feature files.
            return_filename (bool): Whether to return the filename with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
            sample_rate (int): Sampling frequency.
            hop_size (int): Hope size of acoustic feature
            aux_feats (str): Type of auxiliary features.

        """
        # load audio and feature files & check filename
        audio_files = read_txt(to_absolute_path(audio_list))
        feat_files = read_txt(to_absolute_path(feat_list))
        assert check_filename(audio_files, feat_files)

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [load_wav(to_absolute_path(f), sample_rate).shape[0] for f in audio_files]
            idxs = [idx for idx in range(len(audio_files)) if audio_lengths[idx] > audio_length_threshold]
            if len(audio_files) != len(idxs):
                logger.warning(
                    f"Some files are filtered by audio length threshold " f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            feat_files = [feat_files[idx] for idx in idxs]
        if feat_length_threshold is not None:
            f0_lengths = [read_hdf5(to_absolute_path(f), "/f0").shape[0] for f in feat_files]
            idxs = [idx for idx in range(len(feat_files)) if f0_lengths[idx] > feat_length_threshold]
            if len(feat_files) != len(idxs):
                logger.warning(
                    f"Some files are filtered by mel length threshold " f"({len(feat_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            feat_files = [feat_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"${audio_list} is empty."
        assert len(audio_files) == len(
            feat_files
        ), f"Number of audio and features files are different ({len(audio_files)} vs {len(feat_files)})."

        self.audio_files = audio_files
        self.feat_files = feat_files
        self.return_filename = return_filename
        self.allow_cache = allow_cache
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.aux_feats = aux_feats
        self.f0_factor = f0_factor
        self.formants_factor = formants_factor
        logger.info(f"Feature type : {self.aux_feats}")

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

        # define feature pre-processing function
        self.scaler = load(stats)

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_filename = True).
            ndarray: Audio signal (T,).
            ndarray: Auxiliary features (T', C).
            ndarray: F0 sequence (T', 1).
            ndarray: Continuous F0 sequence (T', 1).¥

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]
        # load audio and features
        audio, sr = load_wav(to_absolute_path(self.audio_files[idx]), self.sample_rate)

        # get auxiliary features
        aux_feats = []
        for feat_type in self.aux_feats:
            if feat_type in ["lcf0"]:
                aux_feat = read_hdf5(to_absolute_path(self.feat_files[idx]), f"/{feat_type.replace('l', '')}")
                aux_feat = np.log(aux_feat)
            else:
                aux_feat = read_hdf5(to_absolute_path(self.feat_files[idx]), f"/{feat_type}")
            aux_feat = self.scaler[f"{feat_type}"].transform(aux_feat)
            aux_feats += [aux_feat]
        aux_feats = np.concatenate(aux_feats, axis=1, dtype=np.float32)

        # get dilated factor sequences
        f0 = read_hdf5(to_absolute_path(self.feat_files[idx]), "/f0")  # discrete F0
        cf0 = read_hdf5(to_absolute_path(self.feat_files[idx]), "/cf0")  # continuous F0

        # adjust length
        aux_feats, f0, cf0, audio = validate_length((aux_feats, f0, cf0), (audio,), self.hop_size)

        if self.return_filename:
            items = self.feat_files[idx], audio, aux_feats, f0, cf0
        else:
            items = audio, aux_feats, f0, cf0

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class FeatDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(
        self,
        stats: str,
        feat_list: str,
        feat_length_threshold: Optional[int] = None,
        return_filename: bool = False,
        allow_cache: bool = False,
        sample_rate: int = 24000,
        hop_size: int = 120,
        aux_feats: list[str] = [
            "uv",
            "lcf0",
            "cf1",
            "cf2",
            "cf3",
            "cf4",
            "slope",
            "centroid",
            "energy",
        ],
        f0_factor: float = 1.0,
        formants_factor: list[float] = [1.0, 1.0, 1.0, 1.0],
    ):
        """Initialize dataset.

        Args:
            stats: Filename of the statistic hdf5 file.
            feat_list: Filename of the list of feature files.
            feat_length_threshold: Threshold to remove short feature files.
            return_filename: Whether to return the utterance id with arrays.
            allow_cache: Whether to allow cache of the loaded files.
            sample_rate: Sampling frequency.
            hop_size: Hope size of acoustic feature
            aux_feats: Type of auxiliary features.
            f0_factor: Ratio of scaled f0.
            formants_factor: Ratio of scaled f0.
        """
        # load feat. files
        feat_files = read_txt(to_absolute_path(feat_list))

        # filter by threshold
        if feat_length_threshold is not None:
            f0_lengths = [read_hdf5(to_absolute_path(f), "/f0").shape[0] for f in feat_files]
            idxs = [idx for idx in range(len(feat_files)) if f0_lengths[idx] > feat_length_threshold]
            if len(feat_files) != len(idxs):
                logger.warning(
                    f"Some files are filtered by mel length threshold " f"({len(feat_files)} -> {len(idxs)})."
                )
            feat_files = [feat_files[idx] for idx in idxs]

        # assert the number of files
        assert len(feat_files) != 0, f"${feat_list} is empty."

        self.feat_files = feat_files
        self.return_filename = return_filename
        self.feat_length_threshold = feat_length_threshold
        self.allow_cache = allow_cache
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.aux_feats = aux_feats
        self.f0_factor = f0_factor
        self.formants_factor = formants_factor
        logger.info(f"Feature type : {self.aux_feats}")

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(feat_files))]

        # define feature pre-processing function
        self.scaler = load(stats)

    def __getitem__(self, idx: int) -> tuple:
        """Get specified idx items.

        Args:
            idx: Index of the item.

        Returns:
            str: Utterance id (only in return_filename = True).
            ndarray: Auxiliary feature (T', C).
            ndarray: F0 sequence (T', 1).
            ndarray: Continuous F0 sequence (T', 1).
        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        aux_feats = []
        for feat_type in self.aux_feats:
            if feat_type in ["lcf0"]:
                aux_feat = read_hdf5(to_absolute_path(self.feat_files[idx]), f"/{feat_type.replace('l', '')}")
                aux_feat = np.log(aux_feat)
            else:
                aux_feat = read_hdf5(to_absolute_path(self.feat_files[idx]), f"/{feat_type}")
            aux_feat = self.scaler[f"{feat_type}"].transform(aux_feat)
            if feat_type in ["lcf0"]:
                aux_feat = np.log(np.exp(aux_feat) * self.f0_factor)
            elif feat_type in ["cf1", "cf2", "cf3", "cf4"]:
                aux_feat *= self.formants_factor[int(feat_type[-1]) - 1]
            aux_feats += [aux_feat]
        aux_feats = np.concatenate(aux_feats, axis=1)
        # get dilated factor sequences
        f0 = read_hdf5(to_absolute_path(self.feat_files[idx]), "/f0")  # discrete F0
        cf0 = read_hdf5(to_absolute_path(self.feat_files[idx]), "/cf0")  # continuous F0

        # adjust length
        aux_feats, f0, cf0 = validate_length((aux_feats, f0, cf0))

        if self.return_filename:
            items = self.feat_files[idx], aux_feats, f0, cf0
        else:
            items = aux_feats, f0, cf0

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self) -> int:
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.feat_files)


class MelFeatDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(
        self,
        stats: str,
        feat_list: str,
        feat_length_threshold: Optional[int] = None,
        return_filename: bool = False,
        allow_cache: bool = False,
        sample_rate: int = 24000,
        hop_size: int = 120,
        aux_feats: list[str] = [
            "uv",
            "lcf0",
            "cf1",
            "cf2",
            "cf3",
            "cf4",
            "slope",
            "centroid",
            "energy",
        ],
        f0_factor: float = 1.0,
        formants_factor: list[float] = [1.0, 1.0, 1.0, 1.0],
    ):
        """Initialize dataset.

        Args:
            stats: Filename of the statistic hdf5 file.
            feat_list: Filename of the list of feature files.
            feat_length_threshold: Threshold to remove short feature files.
            return_filename: Whether to return the utterance id with arrays.
            allow_cache: Whether to allow cache of the loaded files.
            sample_rate: Sampling frequency.
            hop_size: Hope size of acoustic feature
            aux_feats: Type of auxiliary features.
            f0_factor: Ratio of scaled f0.
            formants_factor: Ratio of scaled f0.
        """
        # load feat. files
        feat_files = read_txt(to_absolute_path(feat_list))

        # filter by threshold
        if feat_length_threshold is not None:
            f0_lengths = [read_hdf5(to_absolute_path(f), "/f0").shape[0] for f in feat_files]
            idxs = [idx for idx in range(len(feat_files)) if f0_lengths[idx] > feat_length_threshold]
            if len(feat_files) != len(idxs):
                logger.warning(
                    f"Some files are filtered by mel length threshold " f"({len(feat_files)} -> {len(idxs)})."
                )
            feat_files = [feat_files[idx] for idx in idxs]

        # assert the number of files
        assert len(feat_files) != 0, f"${feat_list} is empty."

        self.feat_files = feat_files
        self.return_filename = return_filename
        self.feat_length_threshold = feat_length_threshold
        self.allow_cache = allow_cache
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.aux_feats = aux_feats
        self.f0_factor = f0_factor
        self.formants_factor = formants_factor
        logger.info(f"Feature type : {self.aux_feats}")

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(feat_files))]

        # define feature pre-processing function
        self.scaler = load(stats)

    def __getitem__(self, idx: int) -> tuple:
        """Get specified idx items.

        Args:
            idx: Index of the item.

        Returns:
            str: Utterance id (only in return_filename = True).
            ndarray: Auxiliary feature (T', C).
            ndarray: Mel Spectrogram (T', C).
        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        aux_feats = []
        for feat_type in self.aux_feats:
            if feat_type in ["lcf0"]:
                aux_feat = read_hdf5(to_absolute_path(self.feat_files[idx]), f"/{feat_type.replace('l', '')}")
                aux_feat = np.log(aux_feat)
            else:
                aux_feat = read_hdf5(to_absolute_path(self.feat_files[idx]), f"/{feat_type}")
            aux_feat = self.scaler[f"{feat_type}"].transform(aux_feat)
            if feat_type in ["lcf0"]:
                aux_feat = np.log(np.exp(aux_feat) * self.f0_factor)
            elif feat_type in ["cf1", "cf2", "cf3", "cf4"]:
                aux_feat *= self.formants_factor[int(feat_type[-1]) - 1]
            aux_feats += [aux_feat]
        aux_feats = np.concatenate(aux_feats, axis=1)

        mfbsp = read_hdf5(to_absolute_path(self.feat_files[idx]), "/mfbsp")

        # adjust length
        aux_feats, mfbsp = validate_length((aux_feats, mfbsp))

        if self.return_filename:
            items = self.feat_files[idx], mfbsp, aux_feats
        else:
            items = mfbsp, aux_feats

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self) -> int:
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.feat_files)
