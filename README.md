# RL with different regularizations

## Installation
This repository uses `conda` to manage the environment.
First install `miniconda3` if you do not have it installed.

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```
After installation initialize bash or zsh shells.

```bash
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

Create a new environment and install the required packages.

```bash
conda create --name jax python=3.10
conda activate jax
pip install -r requirements.txt
```

Make sure that `pip` points to the correctly new installed version in `~/miniconda3/envs/jax/bin/pip`.
Double check that `python` points to the correct version in `~/miniconda3/envs/jax/bin/python` and version 3.10.
You can check it by running:
    
```bash
which pip
which python
```

To deactivate the environment run:

```bash
conda deactivate
```

Conda should create an empty environment, but sometimes it does not and copies global packages from
site-packages. This might result in unexpected bugs with packages. To resolve this issue, you can delete all packages from site-packages

## Running the code

### Run locally
If you environment is not activated, activate it by running:

```bash
conda activate jax
```

Run the training code:

```bash
python train_parallel.py --env_name=acrobot-swingup --network_regularization=1 --critic_depth=2
```
The above command is just an example. Check all the possible arguments by analyzing the `train_parallel.py` file.

### Run on a Slurm cluster
If you are using a Slurm cluster, then `*.sh` file in the `./scripts` directory are useful especially
when benchmarking on many GPUs in parallel. Conda's environment will be activated automatically by the script.

Example script run:

```bash
srun --partition=your_partition --qos=your_quality_of_service --gres=gpu:1 ./scripts/acrobot2.sh
```
Probably more flags will be needed depending on the cluster configuration.
Before running the script ensure that they have correct permissions:

```bash
chmod u+x ./scripts/acrobot2.sh
```

### Useful bash scripts
When running the scripts for the first time remember to run:

```bash
chmod u+x <script_name>.sh
```
#### rsync.sh

Rsync sends the code to the cluster.
The script assumes that there exists an environment variable `BBF_DAC_CLUSTER_USERNAME` with your username (can be added in ~/.bashrc - remember to source this file or reopen the terminal)..
If your folder structure on the cluster is different, then change the paths in the script
When used without flags it copies the whole source code to the cluster.

```bash
./rsync.sh
```

When used with flag -n it copies the src but puts it inside an experiment folder with a timestamp.

```bash
./rsync.sh -n
```

#### srun_experiment.sh
Coming soon.

### Hardware requirements
The code can run on a single GPU with 11Gb of memory. In such setup 10 seed for one task can be executed in parallel.
At least 2Gb of RAM memory is required to run the code.