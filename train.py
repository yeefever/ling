"""
Trainining script for Tacotron speech synthesis model.

usage: train.py [options]

options:
    --data-root=<dir>         Directory contains preprocessed features.
    --checkpoint-dir=<dir>    Directory where to save model checkpoints [default: checkpoints].
    --checkpoint-path=<name>  Restore model from checkpoint path if given.
    --hparams=<params>        Hyper parameters [default: ].
    --nepochs=<n>             Number of epochs to train [default: 100].
    -h, --help                Show this help message and exit
"""
from docopt import docopt
import time

# Use text & audio modules from existing Tacotron implementation.
import sys
from os.path import dirname, join
tacotron_lib_dir = join(dirname(__file__), "lib", "tacotron")
sys.path.append(tacotron_lib_dir)
from text import text_to_sequence, symbols
from util import audio
from util.plot import plot_alignment
from tqdm import tqdm, trange
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from hparams import get_hparams

# The tacotron model
from tacotron_pytorch import Tacotron

import torch
from torch.utils import data as data_utils
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from os.path import join, expanduser

import librosa.display
from matplotlib import pyplot as plt
import sys
import os
import tensorboard_logger
from tensorboard_logger import log_value
from hparams import hparams, hparams_debug_string

# Default DATA_ROOT
DATA_ROOT = join("tacotron", "training")

# torch.multiprocessing.set_start_method('spawn')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


fs = hparams.sample_rate

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False


def _pad(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x


class TextDataSource(FileDataSource):
    def __init__(self):
        self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]

    def collect_files(self):
        meta = join(DATA_ROOT, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        lines = list(map(lambda l: l.decode("utf-8").split("|")[-1], lines))
        return lines

    def collect_features(self, text):
        return np.asarray(text_to_sequence(text, self._cleaner_names),
                          dtype=np.int32)


class _NPYDataSource(FileDataSource):
    def __init__(self, col):
        self.col = col

    def collect_files(self):
        meta = join(DATA_ROOT, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        lines = list(map(lambda l: l.decode("utf-8").split(",")[self.col], lines))
        paths = list(map(lambda f: join(DATA_ROOT, f), lines))
        return paths

    def collect_features(self, path):
        return np.load(path)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self):
        super(MelSpecDataSource, self).__init__(1)


class LinearSpecDataSource(_NPYDataSource):
    def __init__(self):
        super(LinearSpecDataSource, self).__init__(0)


class PyTorchDataset(object):
    def __init__(self, X, Mel, Y):
        self.X = X
        self.Mel = Mel
        self.Y = Y

    def __getitem__(self, idx):
        return self.X[idx], self.Mel[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)

def collate_fn(batch):
    """Create batch"""
    r = hparams.outputs_per_step
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = np.max(input_lengths)
    max_target_len = np.max([len(x[1]) for x in batch]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    padded_batch = np.array([_pad(x[0], max_input_len) for x in batch])
    x_batch = torch.from_numpy(padded_batch).long().to(device)

    input_lengths = torch.LongTensor(input_lengths).to(device)

    padded_mel_batch = np.array([_pad_2d(x[1], max_target_len) for x in batch], dtype=np.float32)
    padded_y_batch = np.array([_pad_2d(x[2], max_target_len) for x in batch], dtype=np.float32)

    # Convert directly to PyTorch FloatTensors
    mel_batch = torch.from_numpy(padded_mel_batch).to(device)
    y_batch = torch.from_numpy(padded_y_batch).to(device)

    return x_batch, input_lengths, mel_batch, y_batch


def save_alignment(path, attn):
    plot_alignment(attn.T, path, info="tacotron, step={}".format(global_step))


def save_spectrogram(path, linear_output):
    spectrogram = audio._denormalize(linear_output)
    plt.figure(figsize=(16, 10))
    plt.imshow(spectrogram.T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()


def _learning_rate_decay(init_lr, global_step):
    warmup_steps = 4000.0
    step = global_step + 1.
    lr = init_lr * warmup_steps**0.5 * np.minimum(
        step * warmup_steps**-1.5, step**-0.5)
    return lr


def save_states(global_step, mel_outputs, linear_outputs, attn, y,
                input_lengths, checkpoint_dir=None):
    print("Save intermediate states at step {}".format(global_step))

    # idx = np.random.randint(0, len(input_lengths))
    idx = min(1, len(input_lengths) - 1)
    input_length = input_lengths[idx]

    # Alignment
    path = join(checkpoint_dir, "step{}_alignment.png".format(
        global_step))
    # alignment = attn[idx].cpu().data.numpy()[:, :input_length]
    alignment = attn[idx].cpu().data.numpy()
    save_alignment(path, alignment)

    # Predicted spectrogram
    path = join(checkpoint_dir, "step{}_predicted_spectrogram.png".format(
        global_step))
    linear_output = linear_outputs[idx].cpu().data.numpy()
    save_spectrogram(path, linear_output)

    # Predicted audio signal
    signal = audio.inv_spectrogram(linear_output.T)
    path = join(checkpoint_dir, "step{}_predicted.wav".format(
        global_step))
    audio.save_wav(signal, path)

    # Target spectrogram
    path = join(checkpoint_dir, "step{}_target_spectrogram.png".format(
        global_step))
    linear_output = y[idx].cpu().data.numpy()
    save_spectrogram(path, linear_output)

def train(model, data_loader, optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):
    model.train()
    model = model.to(device)  # Ensure model is on the correct device
    linear_dim = model.linear_dim
    criterion = nn.L1Loss().to(device)  # Loss function on device

    global global_step, global_epoch
    epoch_timer = time.time()
    log_interval = 10
    loss_history = []  # Store loss history for plotting

    while global_epoch < nepochs:
        running_loss = 0.
        for step, (x, input_lengths, mel, y) in enumerate(data_loader):
            # Decay learning rate
            current_lr = _learning_rate_decay(init_lr, global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()

            # Sort by length
            sorted_lengths, indices = torch.sort(
                input_lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long().to(device)

            x, mel, y = x[indices].to(device), mel[indices].to(device), y[indices].to(device)

            # Feed data
            mel_outputs, linear_outputs, attn = model(x, mel, input_lengths=sorted_lengths)

            # Loss
            mel_loss = criterion(mel_outputs, mel)
            n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)
            linear_loss = 0.5 * criterion(linear_outputs, y) \
                + 0.5 * criterion(linear_outputs[:, :, :n_priority_freq],
                                  y[:, :, :n_priority_freq])
            loss = mel_loss + linear_loss

            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_states(
                    global_step, mel_outputs, linear_outputs, attn, y,
                    sorted_lengths.cpu(), checkpoint_dir)
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            # Update
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_thresh)
            optimizer.step()

            # Logs
            log_value("loss", float(loss.item()), global_step)
            log_value("mel loss", float(mel_loss.item()), global_step)
            log_value("linear loss", float(linear_loss.item()), global_step)
            log_value("gradient norm", grad_norm, global_step)
            log_value("learning rate", current_lr, global_step)

            # Timing
            if global_epoch % log_interval == 0:
                epoch_end_time = time.time()
                elapsed_time = epoch_end_time - epoch_timer
                print(f"Epoch {global_epoch}/{nepochs} - Loss: {averaged_loss:.4f} - Time for last {log_interval} epochs: {elapsed_time:.2f}s")
                epoch_timer = epoch_end_time  # Reset timer for next 10 epochs

            global_step += 1
            running_loss += loss.item()

        averaged_loss = running_loss / len(data_loader)
        log_value("loss (per epoch)", averaged_loss, global_epoch)

        # Store the averaged loss for plotting
        loss_history.append(averaged_loss)

        # Plot every 100 epochs
        if global_epoch % 100 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(loss_history, label='Training Loss')
            plt.title(f"Training Loss over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{checkpoint_dir}/loss_plot_epoch_{global_epoch}.png')
            plt.close()

        global_epoch += 1

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{}.pth".format(global_step))
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint-path"]
    data_root = args["--data-root"]
    num_epochs = int(args["--nepochs"])
    if data_root:
        DATA_ROOT = data_root

    # Override hyper parameters
    hparams = get_hparams() 
    print(hparams.batch_size)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Input dataset definitions
    X = FileSourceDataset(TextDataSource())
    Mel = FileSourceDataset(MelSpecDataSource())
    Y = FileSourceDataset(LinearSpecDataSource())

    # Dataset and Dataloader setup
    dataset = PyTorchDataset(X, Mel, Y)
    data_loader = data_utils.DataLoader(
        dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn, pin_memory=hparams.pin_memory)

    # Model
    model = Tacotron(n_vocab=len(symbols),
                     embedding_dim=256,
                     mel_dim=hparams.num_mels,
                     linear_dim=hparams.num_freq,
                     r=hparams.outputs_per_step,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     ).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=hparams.initial_learning_rate, betas=(
                               hparams.adam_beta1, hparams.adam_beta2),
                           weight_decay=hparams.weight_decay)

    # Load checkpoint
    if checkpoint_path:
        print("Load checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        try:
            global_step = checkpoint["global_step"]
            global_epoch = checkpoint["global_epoch"]
        except:
            # TODO
            pass

    # Setup tensorboard logger
    tensorboard_logger.configure("log/run-test")

    # print(hparams_debug_string())

    # Train!
    try:
        train(model, data_loader, optimizer,
              init_lr=hparams.initial_learning_rate,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=num_epochs,
              clip_thresh=hparams.clip_thresh)
    except KeyboardInterrupt:
        save_checkpoint(
            model, optimizer, global_step, checkpoint_dir, global_epoch)


    checkpoint_final = {
        'epoch': global_epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    checkpoint_path = f"final_{global_step}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    print("Finished")
    sys.exit(0)
