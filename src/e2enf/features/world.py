from typing import Optional

import numpy as np
import pyreaper
import pysptk
import pyworld as pw


def sp2mc(sp: np.ndarray, order: int = 24, sr: int = 16000) -> np.ndarray:
    return pysptk.sp2mc(sp, order=order, alpha=pysptk.util.mcepalpha(sr))


def mc2sp(mc: np.ndarray, sr: int = 16000) -> np.ndarray:
    return pysptk.mc2sp(mc, alpha=pysptk.util.mcepalpha(sr))


def code_spenv(sp: np.ndarray, order: int = 24, sr: int = 16000) -> np.ndarray:
    return pw.code_spectral_envelope(sp, sr, order)


def decode_spenv(sp: np.ndarray, sr: int = 16000) -> np.ndarray:
    return pw.decode_spectral_envelope(sp, sr)


def code_ap(ap: np.ndarray, sr: int = 16000) -> np.ndarray:
    return pw.code_aperiodicity(ap, sr)


def decode_ap(ap: np.ndarray, sr: int = 16000) -> np.ndarray:
    return pw.decode_aperiodicity(ap, sr)


class WORLD:
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: Optional[int] = 80,
        f0_min: float = 71.0,
        f0_max: Optional[float] = 800.0,
        f0_extractor: str = "harvest",
        fix_by_reaper: bool = False,
    ):
        self.sample_rate = sample_rate
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_extractor = f0_extractor
        self.fix_by_reaper = fix_by_reaper
        self.hop_length = hop_length
        self.fft_size = pw.get_cheaptrick_fft_size(sample_rate, f0_min)
        self.frame_period = 1000 * hop_length / sample_rate
        if fix_by_reaper:
            self.frame_period /= 2
            self.hop_length = int(self.hop_length / 2)

    def synthesize(self, f0: np.ndarray, sp: np.ndarray, ap: np.ndarray) -> np.ndarray:
        return pw.synthesize(f0, sp, ap, self.frame_period, self.sample_rate)

    def __call__(self, x: np.ndarray, norm: bool = False) -> np.ndarray:
        """calc world features

        Args:
            x: audio signal

        Raises:
            ValueError: f0 method is invalid
            Can use "harvest" or "dio"

        Returns:
            f0: f0 (T, )
            time: time (T, )
            sp: spectral envelope (T x (fft_size // 2 + 1))
            ap: aperiodicity (T x (fft_size // 2 + 1))
        """
        raw_x = x.copy(order="C")
        if norm:
            x = x.astype(np.float64) / np.max(np.abs(x)) * 0.95
        else:
            x = x.astype(np.float64)
        if self.f0_extractor == "harvest":
            f0, time = pw.harvest(
                x, self.sample_rate, f0_floor=self.f0_min, f0_ceil=self.f0_max, frame_period=self.frame_period
            )
        elif self.f0_extractor == "dio":
            f0, time = pw.dio(
                x, self.sample_rate, f0_floor=self.f0_min, f0_ceil=self.f0_max, frame_period=self.frame_period
            )
            f0 = pw.stonemask(x, f0, time, self.sample_rate)
        else:
            raise ValueError(f"Invalid f0 method: {self.f0_extractor}")

        if self.fix_by_reaper:
            if x.dtype != np.int16:
                raw_x = (raw_x * 32768.0).astype(np.int16)
            _, _, _time, f0_mask, _ = pyreaper.reaper(
                raw_x,
                self.sample_rate,
                minf0=self.f0_min,
                maxf0=self.f0_max,
                frame_period=self.frame_period / 1000,
            )
            f0 = f0[1::2].copy(order="C")
            time = time[1::2].copy(order="C")
            f0_mask = f0_mask[1::2]
            f0_mask = np.pad(f0_mask, (0, f0.shape[0] - f0_mask.shape[0]))
            f0 = np.where(f0_mask == -1.0, 0, f0).copy(order="C")
            _time = _time[1::2]
            time[: _time.shape[0]] = _time

        sp = pw.cheaptrick(x, f0, time, self.sample_rate)
        ap = pw.d4c(x, f0, time, self.sample_rate)

        return f0, time, sp, ap


if __name__ == "__main__":
    import librosa

    world = WORLD(f0_extractor="dio")
    world_fixed = WORLD(f0_extractor="dio", fix_by_reaper=True)
    x, sr = librosa.load("./tests/jvs001.wav")
    f0, time, sp, ap = world(x)
    f0_fixed, time_fixed, sp_fixed, ap_fixed = world_fixed(x)
    print(f0.shape, time.shape, sp.shape, ap.shape)
    print(f0_fixed.shape, time_fixed.shape, sp_fixed.shape, ap_fixed.shape)
