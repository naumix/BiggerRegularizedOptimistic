import os

os.environ['MUJOCO_GL'] = 'egl'

import random
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from jaxrl.bro.bro_learner import BRO
from jaxrl.sac.sac_learner import SAC
from jaxrl.replay_buffer import ParallelReplayBuffer
from jaxrl.utils import mute_warning, log_to_wandb_if_time_to, evaluate_if_time_to
from jaxrl.envs.multitask_gym import make_env_mt

import wandb

FLAGS = flags.FLAGS

## DO NOT TOUCH

flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_string('algo', 'BRO', 'Tensorboard logging dir.')
flags.DEFINE_integer('task_type', int(10),'Number of training steps to start training.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 5, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 25000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 128, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1000000), 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', int(1000000), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(2500),'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_boolean('distributional', True, 'Use tqdm progress bar.')


config_flags.DEFINE_config_file('config', 'configs/bro_default.py', 'File path to the training hyperparameter configuration.', lock_config=False)

'''
class flags:
    seed=0
    replay_buffer_size=int(1e6)
    max_steps=int(500)
    start_training=int(10000)
    batch_size=int(128)
    updates_per_step=int(8)
    algo='BRO'
    task_type=int(50)
    
FLAGS = flags()

'''   
    
def main(_):
    save_dir = f'./results/RR10/'
    wandb.init(
        config=FLAGS,
        entity='naumix',
        project='BRO',
        group=f'MT_{FLAGS.task_type}',
        name=f'{FLAGS.algo}_{FLAGS.seed}'
    )
    os.makedirs(save_dir, exist_ok=True)
    env = make_env_mt(FLAGS.seed, FLAGS.task_type)
    eval_env = make_env_mt(FLAGS.seed, FLAGS.task_type)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    mute_warning()

    # Kwargs setup
    all_kwargs = FLAGS.flag_values_dict()
    all_kwargs.update(all_kwargs.pop('config'))
    kwargs = dict(FLAGS.config)
    task_ids = np.eye(FLAGS.task_type)

    if FLAGS.algo == 'BRO':
        updates_per_step = 8
        kwargs['updates_per_step'] = updates_per_step
        kwargs['distributional'] = FLAGS.distributional    
        agent = BRO(
            FLAGS.seed,
            env.observation_space.sample()[0, np.newaxis],
            env.action_space.sample()[0, np.newaxis],
            task_type=FLAGS.task_type,
            num_seeds=1,
            **kwargs,
        )
    else:
        updates_per_step = 1
        kwargs['updates_per_step'] = 1
        kwargs['distributional'] = False  
        agent = SAC(
            FLAGS.seed,
            env.observation_space.sample()[0, np.newaxis],
            env.action_space.sample()[0, np.newaxis],
            task_type=FLAGS.task_type,
            num_seeds=1,
            **kwargs,
        )
        
    replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1], FLAGS.replay_buffer_size, num_seeds=FLAGS.task_type)
    observations = env.reset() 
    eval_returns = [[] for _ in range(FLAGS.task_type)]
    for i in range(1, FLAGS.max_steps+1):
        actions = env.action_space.sample() if i < FLAGS.start_training else agent.sample_actions_o(observations, task_ids, temperature=1.0)
        next_observations, rewards, terms, truns, _ = env.step(actions)
        masks = env.generate_masks(terms, truns)
        replay_buffer.insert(observations, actions, rewards, masks, truns, next_observations, task_ids)
        observations = next_observations
        observations, terms, truns, reward_mask = env.reset_where_done(observations, terms, truns)
        if i >= FLAGS.start_training:
            batches = replay_buffer.sample_parallel_multibatch(128*FLAGS.task_type, updates_per_step)
            infos = agent.update(batches, updates_per_step, i)
            log_to_wandb_if_time_to(i, infos, FLAGS.eval_interval)
        evaluate_if_time_to(i, agent, eval_env, FLAGS.eval_interval, 2, eval_returns, list(range(FLAGS.seed, FLAGS.seed+FLAGS.task_type)), save_dir)
        #eval_env.evaluate(agent, 2)
if __name__ == '__main__':
    app.run(main)
