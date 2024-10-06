import os
import sys
from collections import defaultdict
from logging import getLogger

import hydra
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from e2enf.datasets import MelFeatDataset
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


# A logger for this file
logger = getLogger(__name__)


class Trainer(object):
    """Customized trainer module for NeuralFormants training."""

    def __init__(
        self,
        config: dict,
        steps: int,
        epochs: int,
        data_loader: dict,
        model: dict,
        criterion: dict,
        optimizer: dict,
        scheduler: dict,
        scaler: dict,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            config: Config dict loaded from yaml format configuration file.
            steps: Initial global steps.
            epochs: Initial global epochs.
            data_loader: Dict of data loaders. It must constrain "train" and "dev" loaders.
            model: Dict of models. It must constrain "generator" and "discriminator" models.
            criterion: Dict of criterions. It must constrain "adv", "encode" and "f0" criterions.
            optimizer: Dict of optimizers. It must constrain "generator" and "discriminator" optimizers.
            scheduler: Dict of schedulers. It must constrain "generator" and "discriminator" schedulers.
            device: Pytorch device instance.

        """
        self.config = config
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.use_amp = config.train.use_amp
        self.finish_train = False
        self.writer = SummaryWriter(config.out_dir)
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config.train.train_max_steps, desc="[train]", dynamic_ncols=True
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logger.info("Finished training.")

    def save_checkpoint(self, checkpoint_path: str):
        """Save checkpoint.

        Args:
            checkpoint_path: Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "neuralformants": self.optimizer["neuralformants"].state_dict(),
            },
            "scheduler": {
                "neuralformants": self.scheduler["neuralformants"].state_dict(),
            },
            "scaler": {
                "neuralformants": self.scaler["neuralformants"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = {
            "neuralformants": self.model["neuralformants"].state_dict(),
        }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str, load_only_params: bool = False):
        """Load checkpoint.

        Args:
            checkpoint_path: Checkpoint path to be loaded.
            load_only_params: Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model["neuralformants"].load_state_dict(state_dict["model"]["neuralformants"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["neuralformants"].load_state_dict(state_dict["optimizer"]["neuralformants"])
            self.scheduler["neuralformants"].load_state_dict(state_dict["scheduler"]["neuralformants"])
            self.scaler["neuralformants"].load_state_dict(state_dict["scaler"]["neuralformants"])

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        y, x = batch
        y, x = y.to(self.device), x.to(self.device)

        # generator forward
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            y_hat = self.model["neuralformants"](x)

            # calculate spectral loss
            loss = self.criterion["loss"](y, y_hat)
            self.total_train_loss["train/mse_loss"] += loss.item()

        # update generator
        self.optimizer["neuralformants"].zero_grad()
        self.scaler["neuralformants"].scale(loss).backward()
        self.scaler["neuralformants"].unscale_(self.optimizer["neuralformants"])
        if self.config.train.grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["neuralformants"].parameters(),
                self.config.train.grad_norm,
            )
        self.scaler["neuralformants"].step(self.optimizer["neuralformants"])
        self.scaler["neuralformants"].update()
        self.scheduler["neuralformants"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            self._check_log_interval()
            self._check_eval_interval()
            self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                self._check_save_interval(force=True)
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        y, x = batch
        y, x = y.to(self.device), x.to(self.device)

        # generator forward
        y_hat = self.model["neuralformants"](x)

        # calculate spectral loss
        loss = self.criterion["loss"](y, y_hat)
        self.total_train_loss["eval/mse_loss"] += loss.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        # logger.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.data_loader["valid"], desc="[eval]"), 1):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)
            if eval_steps_per_epoch == 3:
                break

        # logger.info(f"(Steps: {self.steps}) Finished evaluation " f"({eval_steps_per_epoch} steps per epoch).")

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            # logger.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # parse batch
        _, x = batch
        x = x.to(self.device)

        # generator forward
        y_hat = self.model["neuralformants"](x)
        for i, _y_hat in enumerate(y_hat):
            _y_hat = _y_hat.squeeze(0)
            _y_hat = librosa.amplitude_to_db(np.exp(_y_hat.cpu().numpy()), ref=np.max)
            # _y_hat = _y_hat.cpu().numpy()
            fig = plt.figure(figsize=(8, 6))
            librosa.display.specshow(
                _y_hat,
                y_axis="mel",
                x_axis="time",
                sr=self.config.data.sample_rate,
                hop_length=self.config.data.hop_size,
                cmap="viridis",
            )
            # plt.ylim([spectrogram.ymin, spectrogram.ymax])
            plt.xlabel("time [s]")
            plt.ylabel("frequency [Hz]")
            self.writer.add_figure(f"spectrogram/gen_{i}", fig, self.steps)
            plt.clf()
            plt.close()

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self, force=False):
        if force or self.steps % self.config.train.save_interval_steps == 0:
            self.save_checkpoint(
                os.path.join(
                    self.config.out_dir,
                    "checkpoints",
                    f"checkpoint-{self.steps}steps.pkl",
                )
            )
            logger.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config.train.eval_interval_steps == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config.train.log_interval_steps == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config.train.log_interval_steps
                # logger.info(f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.")
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config.train.train_max_steps:
            self.finish_train = True


class Collator(object):
    """Customized collator for Pytorch DataLoader in training."""

    def __init__(
        self,
        batch_max_frames=128,
        sample_rate=24000,
        hop_size=120,
    ):
        """Initialize customized collator for PyTorch DataLoader.

        Args:
            batch_max_length (int): The maximum length of batch.
            sample_rate (int): Sampling rate.
            hop_size (int): Hop size of auxiliary features.
        """
        self.batch_max_frames = batch_max_frames
        self.sample_rate = sample_rate
        self.hop_size = hop_size

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tensor: Auxiliary feature batch (B, C, T').
            Tensor: Target Mel Feature (B, C, T).

        """
        # time resolution check
        y_batch, c_batch = [], []
        for idx in range(len(batch)):
            x, c = batch[idx]  # audio, aux_feats, f0, cf0
            if len(c) > self.batch_max_frames:
                # randomly pickup with the batch_max_length length of the part
                start_frame = np.random.randint(0, len(c) - self.batch_max_frames)
                y = x[start_frame : start_frame + self.batch_max_frames]
                c = c[start_frame : start_frame + self.batch_max_frames]
                self._check_length(y, c)
            else:
                logger.warn(f"Removed short sample from batch (length={len(x)}).")
                continue
            y_batch += [y.astype(np.float32)]  # [(T, D), ...]
            c_batch += [c.astype(np.float32)]  # [(T', D), ...]

        # convert each batch to tensor, asuume that each item in batch has the same length
        y_batch = torch.FloatTensor(np.array(y_batch)).transpose(2, 1)  # (B, 1, T)
        c_batch = torch.FloatTensor(np.array(c_batch)).transpose(2, 1)  # (B, 1, T')

        return y_batch, c_batch

    def _check_length(self, x, c):
        """Assert the audio and feature lengths are correctly adjusted for upsamping."""
        assert len(x) == len(c)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config: DictConfig) -> None:
    """Run training process."""

    if config.device is not None:
        print(f"Device: {config.device}")
        device = torch.device(config.device)
    elif not torch.cuda.is_available():
        print("CPU")
        device = torch.device("cpu")
    else:
        print("GPU")
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True

    # fix seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # check directory existence
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    # write config to yaml file
    with open(os.path.join(config.out_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config))
    logger.info(OmegaConf.to_yaml(config))

    if config.data.remove_short_samples:
        feat_length_threshold = config.data.batch_max_frames
    else:
        feat_length_threshold = None

    train_dataset = MelFeatDataset(
        stats=to_absolute_path(config.data.stats),
        feat_list=to_absolute_path(config.data.train_feat),
        feat_length_threshold=feat_length_threshold,
        allow_cache=config.data.allow_cache,
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
        aux_feats=config.data.aux_feats,
    )
    logger.info(f"The number of training files = {len(train_dataset)}.")

    valid_dataset = MelFeatDataset(
        stats=to_absolute_path(config.data.stats),
        feat_list=to_absolute_path(config.data.valid_feat),
        feat_length_threshold=feat_length_threshold,
        allow_cache=config.data.allow_cache,
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
        aux_feats=config.data.aux_feats,
    )
    logger.info(f"The number of validation files = {len(valid_dataset)}.")

    dataset = {"train": train_dataset, "valid": valid_dataset}

    # get data loader
    collator = Collator(
        batch_max_frames=config.data.batch_max_frames,
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
    )

    train_sampler, valid_sampler = None, None
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=True,
            collate_fn=collator,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            sampler=train_sampler,
            pin_memory=config.data.pin_memory,
        ),
        "valid": DataLoader(
            dataset=dataset["valid"],
            shuffle=True,
            collate_fn=collator,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            sampler=valid_sampler,
            pin_memory=config.data.pin_memory,
        ),
    }

    # define models and optimizers
    model = {
        "neuralformants": hydra.utils.instantiate(config.model).to(device),
    }

    # define training criteria
    criterion = {
        "loss": hydra.utils.instantiate(config.train.loss).to(device),
    }

    # # define optimizers and schedulers
    optimizer = {
        "neuralformants": hydra.utils.instantiate(
            config.train.neuralformants_optimizer, params=model["neuralformants"].parameters()
        ),
    }
    scheduler = {
        "neuralformants": hydra.utils.instantiate(
            config.train.neuralformants_scheduler, optimizer=optimizer["neuralformants"]
        ),
    }
    scaler = {"neuralformants": torch.cuda.amp.GradScaler(enabled=config.train.use_amp)}

    # define trainer
    trainer = Trainer(
        config=config,
        steps=0,
        epochs=0,
        data_loader=data_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )

    # load trained parameters from checkpoint
    if config.train.resume:
        resume = os.path.join(config.out_dir, "checkpoints", f"checkpoint-{config.train.resume}steps.pkl")
        if os.path.exists(resume):
            trainer.load_checkpoint(resume)
            logger.info(f"Successfully resumed from {resume}.")
        else:
            logger.info(f"Failed to resume from {resume}.")
            sys.exit(0)
    else:
        logger.info("Start a new training process.")

    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(os.path.join(config.out_dir, "checkpoints", f"checkpoint-{trainer.steps}steps.pkl"))
        logger.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
