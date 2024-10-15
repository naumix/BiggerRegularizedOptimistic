import os
import numpy as np
import wandb
import logging
import tensorflow_probability.substrates.numpy as tfp

def mute_warning():
    tfp.distributions.TransformedDistribution(tfp.distributions.Normal(0.0, 1.0), tfp.bijectors.Identity())
    logger = logging.getLogger('root')
    class CheckTypesFilter(logging.Filter):
        def filter(self, record):
            return 'check_types' not in record.getMessage()
    logger.addFilter(CheckTypesFilter())

def log_to_wandb_if_time_to1(step, infos, eval_interval, suffix: str = ''):
    if step % eval_interval == 0:
        dict_to_log = {'timestep': step}
        for info_key in infos:
            for seed, value in enumerate(infos[info_key]):
                dict_to_log[f'seed{seed}/{info_key}{suffix}'] = value
        wandb.log(dict_to_log, step=step)
        
def log_to_wandb_if_time_to(step, infos, eval_interval, suffix: str = ''):
    if step % eval_interval == 0:
        dict_to_log = {'timestep': step}
        for info_key in infos:
            dict_to_log[f'seed/{info_key}{suffix}'] = infos[info_key]
        wandb.log(dict_to_log, step=step)

def evaluate_if_time_to(i, agent, eval_env, eval_interval, eval_episodes, eval_returns, seeds, save_dir):
    if i % eval_interval == 0:
        eval_stats = eval_env.evaluate(agent, num_episodes=eval_episodes, temperature=0.0)
        for j, seed in enumerate(seeds):
            eval_returns[j].append((i, eval_stats['return'][j]))
            np.savetxt(os.path.join(save_dir, f'{seed}.txt'), eval_returns[j], fmt=['%d', '%.1f'])
        log_to_wandb_if_time_to1(i, eval_stats, eval_interval, suffix='_eval')
        
def make_env(benchmark, env_name, seed, num_envs):
    if benchmark == 'dmc':
        from jaxrl.envs.dmc_gym import make_env_dmc
        return make_env_dmc(env_name, seed=seed, num_envs=num_envs)
    if benchmark == 'mw':
        from jaxrl.envs.metaworld_gym import make_env_mw
        return make_env_mw(env_name, seed=seed, num_envs=num_envs)
    if benchmark == 'myo':
        from jaxrl.envs.myosuite_gym import make_env_myo
        return make_env_myo(env_name, seed=seed, num_envs=num_envs)
    if benchmark == 'gym':
        from jaxrl.envs.openai_gym import make_env_gym
        return make_env_gym(env_name, seed=seed, num_envs=num_envs)
    if benchmark == 'adroit':
        from jaxrl.envs.adroit_gym import make_env_adroit
        return make_env_adroit(env_name, seed=seed, num_envs=num_envs)
    if benchmark == 'dexhand':
        from jaxrl.envs.dexhand_gym import make_env_dexhand
        return make_env_dexhand(env_name, seed=seed, num_envs=num_envs)

        