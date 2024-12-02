import tensorflow as tf
from collections import namedtuple
from argparse import Namespace

def get_hparams():
  hparams = Namespace(
      cleaners='english_cleaners',
      use_cmudict=False,

      # Audio:
      num_mels=80,
      num_freq=1025,
      sample_rate=20000,
      frame_length_ms=50,
      frame_shift_ms=12.5,
      preemphasis=0.97,
      min_level_db=-100,
      ref_level_db=20,

      # Model:
      outputs_per_step=5,
      padding_idx=None,
      use_memory_mask=False,

      # Data loader
      pin_memory=False,
      num_workers=0,

      # Training:
      batch_size=32,
      adam_beta1=0.9,
      adam_beta2=0.999,
      initial_learning_rate=2e-5,
      decay_learning_rate=True,
      nepochs=1000,
      weight_decay=0.0,
      clip_thresh=1.0,

      # Save
      checkpoint_interval=5000,

      # Eval:
      max_iters=200,
      griffin_lim_iters=60,
      power=1.5,
  )
  return hparams

hparams = get_hparams()


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
