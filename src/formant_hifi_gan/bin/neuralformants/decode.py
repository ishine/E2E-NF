import os
import re
from logging import getLogger
from time import time

import hydra
import numpy as np
import soundfile as sf
import torch
from formant_hifi_gan.datasets import MelFeatDataset
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

# A logger for this file
logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="decode")
def main(config: DictConfig) -> None:
    """Run decoding process."""

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # set device
    if config.device != "":
        device = torch.device(config.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Decode on {device}.")

    # load pre-trained model from checkpoint file
    if config.checkpoint_path is None:
        checkpoint_steps = config.checkpoint_steps
        checkpoint_path = os.path.join(
            config.out_dir,
            "checkpoints",
            f"checkpoint-{config.checkpoint_steps}steps.pkl",
        )
    else:
        pattern = re.compile("\d+")
        m = pattern.search(config.checkpoint_path)
        checkpoint_steps = m.group() if m is not None else "unknown"
        checkpoint_path = config.checkpoint_path
    state_dict = torch.load(to_absolute_path(checkpoint_path), map_location="cpu")
    logger.info(f"Loaded model parameters from {checkpoint_path}.")
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(state_dict["model"]["neuralformants"])
    model.remove_weight_norm()
    model.eval().to(device)

    vocoder_state_dict = torch.load(to_absolute_path(config.vocoder_checkpoint_path), map_location="cpu")
    vocoder = hydra.utils.instantiate(config.vocoder)
    vocoder.load_state_dict(vocoder_state_dict["model"]["generator"])
    vocoder.remove_weight_norm()
    vocoder.eval().to(device)

    # check directory existence
    out_dir = to_absolute_path(os.path.join(config.out_dir, "wav", str(checkpoint_steps)))
    os.makedirs(out_dir, exist_ok=True)

    for f0_factor in config.f0_factors:
        for formants_factor in config.formants_factors:
            dataset = MelFeatDataset(
                stats=to_absolute_path(config.data.stats),
                feat_list=to_absolute_path(config.data.eval_feat),
                allow_cache=config.data.allow_cache,
                sample_rate=config.data.sample_rate,
                hop_size=config.data.hop_size,
                aux_feats=config.data.aux_feats,
                f0_factor=f0_factor,
                formants_factor=formants_factor,
                return_filename=True,
            )
            logger.info(f"The number of features to be decoded = {len(dataset)}.")

            with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
                total_rtf = 0.0
                for idx, (feat_path, _, x) in enumerate(pbar, 1):
                    x = torch.FloatTensor(x.T).unsqueeze(0).to(device)
                    start = time()
                    y = model(x).squeeze(0).cpu().numpy()
                    y = dataset.scaler["mfbsp"].transform(y.T).T
                    outs = vocoder(x=None, c=torch.from_numpy(y).float().unsqueeze(0).to(device))
                    audio = outs[0].squeeze()
                    rtf = (time() - start) / (audio.size(-1) / config.data.sample_rate)
                    pbar.set_postfix({"RTF": rtf})
                    total_rtf += rtf

                    # save output signal as PCM 16 bit wav file
                    utt_id = os.path.splitext(os.path.basename(feat_path))[0]
                    fo = "_".join([f"{f:.2f}" for f in formants_factor])
                    spk_id = feat_path.split(os.path.sep)[config.spkidx]
                    save_dir = os.path.join(out_dir, spk_id)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{utt_id}_f{f0_factor:.2f}_fo{fo}.wav")
                    audio = audio.view(-1).cpu().numpy()
                    sf.write(save_path, audio, config.data.sample_rate, "PCM_16")

                # report average RTF
                mean_rtf = total_rtf / len(dataset)
                logger.info(f"Finished generation of {idx} utterances (RTF: {mean_rtf:.6f}, ×{1 / mean_rtf:.3f}).")


if __name__ == "__main__":
    main()
