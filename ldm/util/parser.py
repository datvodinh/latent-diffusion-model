import argparse


def get_training_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', '-dd', type=str, default=None,
        help='model name'
    )
    parser.add_argument(
        '--dataset', '-ds', type=str, default='celebahq',
        help='model name'
    )
    parser.add_argument(
        '--stage', '-st', type=str, default='stage2',
        help='training state for LDM'
    )
    parser.add_argument(
        '--vae_ckpt', '-vc', type=str, default=None,
        help='checkpoint for pretrained VAE model'
    )
    parser.add_argument(
        '--model_ckpt', '-mc', type=str, default=None,
        help='checkpoint for model for continue training'
    )
    parser.add_argument(
        '--max_epochs', '-me', type=int, default=100,
        help='max epoch'
    )
    parser.add_argument(
        '--batch_size', '-bs', type=int, default=None,
        help='batch size'
    )
    parser.add_argument(
        '--train_ratio', '-tr', type=float, default=None,
        help='batch size'
    )
    parser.add_argument(
        '--val_ratio', '-vr', type=float, default=None,
        help='batch size'
    )
    parser.add_argument(
        '--timesteps', '-ts', type=int, default=None,
        help='max timesteps diffusion'
    )
    parser.add_argument(
        '--max_batch_size', '-mbs', type=int, default=32,
        help='max batch size'
    )
    parser.add_argument(
        '--lr', '-l', type=float, default=None,
        help='learning rate'
    )
    parser.add_argument(
        '--num_workers', '-nw', type=int, default=None,
        help='number of workers'
    )
    parser.add_argument(
        '--seed', '-s', type=int, default=None,
        help='seed'
    )
    parser.add_argument(
        '--name', '-n', type=str, default=None,
        help='name of the experiment'
    )
    parser.add_argument(
        '--pbar', action='store_true',
        help='progress bar'
    )
    parser.add_argument(
        '--precision', '-p', type=str, default='16-mixed',
        help='numerical precision'
    )
    parser.add_argument(
        '--sample_per_epochs', '-spe', type=int, default=None,
        help='sample every n epochs'
    )
    parser.add_argument(
        '--n_samples', '-ns', type=int, default=None,
        help='number of workers'
    )
    parser.add_argument(
        '--wandb', '-wk', type=str, default=None,
        help='wandb API key'
    )
    return parser
