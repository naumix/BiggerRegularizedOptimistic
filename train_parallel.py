import os

os.environ['MUJOCO_GL'] = 'egl'

import random
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from jaxrl.bro.bro_learner import BRO
from jaxrl.replay_buffer import ParallelReplayBuffer
from jaxrl.utils import mute_warning, log_to_wandb_if_time_to, evaluate_if_time_to, make_env

import wandb

FLAGS = flags.FLAGS

## DO NOT TOUCH

flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 5, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 25000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 128, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1000000), 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', int(1000000), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(2500),'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_integer('num_seeds', 10, 'Number of parallel seeds to run.')
flags.DEFINE_integer('updates_per_step', 10, 'Number of updates per step.')

flags.DEFINE_string('benchmark', 'dmc', 'Environment name.')
flags.DEFINE_string('env_name', 'cheetah-run', 'Environment name.')
flags.DEFINE_boolean('distributional', True, 'Use tqdm progress bar.')


config_flags.DEFINE_config_file('config', 'configs/bro_default.py', 'File path to the training hyperparameter configuration.', lock_config=False)

def main(_):
    save_dir = f'./results/{FLAGS.env_name}_RR{str(FLAGS.updates_per_step)}/'
    wandb.init(
        config=FLAGS,
        entity='naumix',
        project='BRO',
        group=f'{FLAGS.env_name}',
        name=f'BRO_Quantile:{FLAGS.distributional}_BS:{FLAGS.batch_size}_RR:{FLAGS.updates_per_step}'
    )
    os.makedirs(save_dir, exist_ok=True)
    env = make_env(FLAGS.benchmark, FLAGS.env_name, FLAGS.seed, num_envs=FLAGS.num_seeds)
    eval_env = make_env(FLAGS.benchmark, FLAGS.env_name, FLAGS.seed + 42, num_envs=FLAGS.num_seeds)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    mute_warning()

    # Kwargs setup
    all_kwargs = FLAGS.flag_values_dict()
    all_kwargs.update(all_kwargs.pop('config'))
    kwargs = dict(FLAGS.config)
    kwargs['updates_per_step'] = FLAGS.updates_per_step
    kwargs['distributional'] = FLAGS.distributional

    agent = BRO(
        FLAGS.seed,
        env.observation_space.sample()[0, np.newaxis],
        env.action_space.sample()[0, np.newaxis],
        num_seeds=FLAGS.num_seeds,
        **kwargs,
    )
    replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1], FLAGS.replay_buffer_size, num_seeds=FLAGS.num_seeds)
    observations = env.reset()
    eval_returns = [[] for _ in range(FLAGS.num_seeds)]
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        actions = env.action_space.sample() if i < FLAGS.start_training else agent.sample_actions_o(observations, temperature=1.0)
        next_observations, rewards, terms, truns, _ = env.step(actions)
        masks = env.generate_masks(terms, truns)
        replay_buffer.insert(observations, actions, rewards, masks, truns, next_observations)
        observations = next_observations
        observations, terms, truns, reward_mask = env.reset_where_done(observations, terms, truns)
        if i >= FLAGS.start_training:
            batches = replay_buffer.sample_parallel_multibatch(FLAGS.batch_size, FLAGS.updates_per_step)
            infos = agent.update(batches, FLAGS.updates_per_step, i)
            log_to_wandb_if_time_to(i, infos, FLAGS.eval_interval)
        evaluate_if_time_to(i, agent, eval_env, FLAGS.eval_interval, FLAGS.eval_episodes, eval_returns, list(range(FLAGS.seed, FLAGS.seed+FLAGS.num_seeds)), save_dir)


if __name__ == '__main__':
    app.run(main)
