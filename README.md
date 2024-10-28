# Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control

https://arxiv.org/abs/2405.16158

This branch contains the implementation of the BRO algorithm (NeurIPS 2024 spotlight) that has been migrated to use the newest versions of mujoco and gymansium. The codebase is heavily inspired by [JaxRL](https://github.com/ikostrikov/jaxrl) and [Parallel JaxRL](https://github.com/proceduralia/high_replay_ratio_continuous_control).

## Example usage

To run the BRO algorithm:
`python3 train_parallel.py --benchmark=dmc --env_name=dog-run --num_seeds=10 --updates_per_step=10`

To run the BRO (fast) version simply reduce the replay ratio to 2:
`python3 train_parallel.py --benchmark=dmc --env_name=dog-run --num_seeds=10 --updates_per_step=2`

## Citation

If you find this repository useful, feel free to cite our paper using the following bibtex.

```
@inproceedings{
nauman2024bigger,
title={Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control},
author={Michal Nauman and Mateusz Ostaszewski and Krzysztof Jankowski and Piotr Miłoś and Marek Cygan},
booktitle={Advances in Neural Information Processing Systems},
year={2024},
url={https://arxiv.org/pdf/2405.16158},
}
```
