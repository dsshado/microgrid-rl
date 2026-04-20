import argparse
import torch


def get_training_config(args=None):
    parser = argparse.ArgumentParser(description="MAPPO/PPO with GCAPS for 123-bus outage management")

    # Algorithm
    parser.add_argument('--algo', type=str, default='MAPPO', choices=['PPO', 'MAPPO'])

    # Network
    parser.add_argument('--features_dim', type=int, default=128)

    # Training
    parser.add_argument('--total_steps',   type=int,   default=500000)
    parser.add_argument('--batch_size',    type=int,   default=2000)
    parser.add_argument('--n_steps',       type=int,   default=50000)
    parser.add_argument('--n_epochs',      type=int,   default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--ent_coef',      type=float, default=0.1)
    parser.add_argument('--val_coef',      type=float, default=0.5)
    parser.add_argument('--gamma',         type=float, default=0.99)

    # Saving & Logging
    parser.add_argument('--save_freq',  type=int, default=100000)
    parser.add_argument('--logger',     type=str, default='Tensorboard_logger/')
    parser.add_argument('--model_save', type=str, default='Trained_Models/')

    # Hardware
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--num_cpu', type=int, default=1)

    config          = parser.parse_args(args)
    config.use_cuda = torch.cuda.is_available() and not config.no_cuda
    return config
