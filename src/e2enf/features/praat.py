import numpy as np
import parselmouth


def fix_formants(formants: np.ndarray) -> np.ndarray:
    """
    value is 0 if the formant is not detected.
    """
    new_formants = np.zeros_like(formants)
    notnan_indices = ~np.isnan(formants)
    new_formants[notnan_indices] = formants[notnan_indices]

    return new_formants


def get_formants(
    y: np.ndarray,
    sr: int = 16000,
    n_formant: float = 5.0,
    hop_length: int = 80,
    win_size: int = 400,
    fmax: float = 5000.0,
    pre_enphasis: float = 50.0,
) -> np.ndarray:
    # actual win_size is win_size * 2, because praat uses a Gaussian-like analysis window with sidelobes below -120dB
    # more details: https://www.fon.hum.uva.nl/praat/manual/Sound__to_Formant__burg____.html
    pad_left = (win_size * 2 - hop_length) // 2
    pad_right = max((win_size * 2 - hop_length + 1) // 2, win_size - y.shape[-1] - pad_left)
    if pad_right < y.shape[-1]:
        mode = "reflect"
    else:
        mode = "constant"
    y = np.pad(y, (pad_left, pad_right), mode=mode)

    time_step = hop_length / sr
    window_length = win_size / sr
    burg_obj = parselmouth.Sound(y, sampling_frequency=sr).to_formant_burg(
        time_step=time_step,
        max_number_of_formants=n_formant,
        maximum_formant=fmax,
        window_length=window_length,
        pre_emphasis_from=pre_enphasis,
    )
    time_from_frame = [burg_obj.frame_number_to_time(frame + 1) for frame in range(burg_obj.n_frames)]
    formants = np.zeros((n_formant, len(time_from_frame)))
    for fn in range(1, n_formant):
        for frame, time in enumerate(time_from_frame):
            formants[fn - 1, frame] = burg_obj.get_value_at_time(fn, time=time, unit="HERTZ")
        # formants[fn - 1, np.isnan(formants[fn - 1])] = 0
    time_from_frame = np.array(time_from_frame).reshape(-1)

    return formants, time_from_frame


# From https://github.com/ChristopherCarignan/formant-optimization
def get_optimized_formants(
    y: np.ndarray,
    sr: int = 16000,
    hop_length: int = 80,
    win_size: int = 400,
    pre_enphasis: float = 50.0,
    search_fo_min: int = 3500,
    search_fo_max: int = 6000,
    search_step: float = 50,
):
    steps = int((search_fo_max - search_fo_min) / search_step)
    n_formant = 5

    formants_baseline, times = get_formants(
        y,
        sr=sr,
        n_formant=n_formant,
        hop_length=hop_length,
        win_size=win_size,
        fmax=search_fo_min,
        pre_enphasis=pre_enphasis,
    )

    f1s = np.zeros((steps + 1, formants_baseline.shape[-1]))
    f2s = np.zeros((steps + 1, formants_baseline.shape[-1]))
    f3s = np.zeros((steps + 1, formants_baseline.shape[-1]))
    f4s = np.zeros((steps + 1, formants_baseline.shape[-1]))
    f5s = np.zeros((steps + 1, formants_baseline.shape[-1]))

    f1s[0] = formants_baseline[0]
    f2s[0] = formants_baseline[1]
    f3s[0] = formants_baseline[2]
    f4s[0] = formants_baseline[3]
    f5s[0] = formants_baseline[4]
    min_length = formants_baseline.shape[-1]

    idx = 1

    for i in range(1, steps + 1):
        step = i * 50
        ceiling = search_fo_min + step

        formants, _ = get_formants(
            y,
            sr=sr,
            n_formant=n_formant,
            hop_length=hop_length,
            win_size=win_size,
            fmax=ceiling,
            pre_enphasis=pre_enphasis,
        )
        length = formants.shape[-1]
        if length >= min_length:
            formants = formants[:, :min_length]
        else:
            min_length = length
            f1s = f1s[:, :length]
            f2s = f2s[:, :length]
            f3s = f3s[:, :length]
            f4s = f4s[:, :length]
            f5s = f5s[:, :length]

        f1s[idx] = formants[0]
        f2s[idx] = formants[1]
        f3s[idx] = formants[2]
        f4s[idx] = formants[3]
        f5s[idx] = formants[4]

        idx += 1

    times = times[:min_length]
    optimized = np.zeros((5, times.shape[0]))

    for i in range(times.shape[0]):
        for j in range(1, 6):
            target_formants = locals()[f"f{j}s"][:, i]
            target_formants[np.isnan(target_formants)] = 0
            diff = np.zeros(steps)
            for k in range(steps):
                diff[k] = np.abs(target_formants[k + 1] - target_formants[k])
            maxidx = np.argmax(diff)

            ftrim = target_formants.copy(order="C")[maxidx:]
            diff = np.zeros(steps - maxidx)
            for k in range(steps - maxidx):
                diff[k] = np.abs(ftrim[k + 1] - ftrim[k])
            minidx = np.argmax(0 - diff)

            optimized[j - 1, i] = ftrim[minidx]

    return optimized, times
