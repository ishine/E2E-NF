import sys
from logging import getLogger

import numpy as np
import torch
from torch.nn.functional import interpolate

# A logger for this file
logger = getLogger(__name__)


class SignalGenerator:
    """Input signal generator module."""

    def __init__(
        self,
        sample_rate=24000,
        hop_size=120,
        sine_amp=0.1,
        noise_amp=0.003,
        signal_types=["sine", "noise"],
    ):
        """

        Args:
            sample_rate (int): Sampling rate.
            hop_size (int): Hop size of input F0.
            sine_amp (float): Sine amplitude for NSF-based sine generation.
            noise_amp (float): Noise amplitude for NSF-based sine generation.
            signal_types (list): List of input signal types for generator.

        """
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.signal_types = signal_types
        self.sine_amp = sine_amp
        self.noise_amp = noise_amp

        for signal_type in signal_types:
            if signal_type not in ["noise", "sine", "sines", "uv"]:
                logger.info(f"{signal_type} is not supported type for generator input.")
                sys.exit(0)
        logger.info(f"Use {signal_types} for generator input signals.")

    @torch.no_grad()
    def __call__(self, f0):
        signals = []
        for typ in self.signal_types:
            if "noise" == typ:
                signals.append(self.random_noise(f0))
            if "sine" == typ:
                signals.append(self.sinusoid(f0))
            if "sines" == typ:
                signals.append(self.sinusoids(f0))
            if "uv" == typ:
                signals.append(self.vuv_binary(f0))

        input_batch = signals[0]
        for signal in signals[1:]:
            input_batch = torch.cat([input_batch, signal], axis=1)
        input_batch = input_batch.float()

        return input_batch

    @torch.no_grad()
    def random_noise(self, f0):
        """Calculate noise signals.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: Gaussian noise signals (B, 1, T).

        """
        B, _, T = f0.size()
        noise = torch.randn((B, 1, T * self.hop_size), device=f0.device)

        return noise

    @torch.no_grad()
    def sinusoid(self, f0):
        """Calculate sine signals.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: Sines generated following NSF (B, 1, T).

        """
        B, _, T = f0.size()
        vuv = interpolate((f0 > 0) * torch.ones_like(f0), T * self.hop_size)
        radious = (interpolate(f0.to(torch.float64), T * self.hop_size) / self.sample_rate) % 1
        sine = vuv * torch.sin(torch.cumsum(radious, dim=2) * 2 * np.pi) * self.sine_amp
        if self.noise_amp > 0:
            noise_amp = vuv * self.noise_amp + (1.0 - vuv) * self.noise_amp / 3.0
            noise = torch.randn((B, 1, T * self.hop_size), device=f0.device) * noise_amp
            sine = sine + noise

        return sine

    @torch.no_grad()
    def sinusoids(self, f0):
        """Calculate sines.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: Sines generated following NSF (B, 1, T).

        """
        B, _, T = f0.size()
        vuv = interpolate((f0 > 0) * torch.ones_like(f0), T * self.hop_size)
        f0 = interpolate(f0.to(torch.float64), T * self.hop_size)
        sines = torch.zeros_like(f0, device=f0.device)
        harmonics = 5  # currently only fixed number of harmonics is supported
        for i in range(harmonics):
            radious = (f0 * (i + 1) / self.sample_rate) % 1
            sines += torch.sin(torch.cumsum(radious, dim=2) * 2 * np.pi)
        sines = self.sine_amp * sines * vuv / harmonics
        if self.noise_amp > 0:
            noise_amp = vuv * self.noise_amp + (1.0 - vuv) * self.noise_amp / 3.0
            noise = torch.randn((B, 1, T * self.hop_size), device=f0.device) * noise_amp
            sines = sines + noise

        return sines

    @torch.no_grad()
    def vuv_binary(self, f0):
        """Calculate V/UV binary sequences.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: V/UV binary sequences (B, 1, T).

        """
        _, _, T = f0.size()
        uv = interpolate((f0 > 0) * torch.ones_like(f0), T * self.hop_size)

        return uv
