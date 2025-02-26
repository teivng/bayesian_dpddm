# Bayesian D-PDDM

Bayesian implementation of the D-PDDM algorithm for post-deployment deterioration monitoring. Bayesian D-PDDM is a Bayesian approximation to the D-PDDM algorithm which provably monitors model deterioration at deployment time. Bayesian D-PDDM:

- Flags deteriorating shifts in the unsupervised deployment data distribution
- Resists flagging non-deteriorating shifts, unlike classical OOD detection leveraging distances and/or metrics between data distributions. 

## Installation and Requirements

This implementation requires ``python>=3.11``. 

The easiest way to install ``bayesian_dpddm`` is with ``pip``:

``pip install bayesian_dpddm``

You can also install by cloning the GitHub repo:

```
# Clone the repo
git clone https://github.com/teivng/bayesian_dpddm.git

# Navigate into repo directory 
cd bayesian_dpddm

# Install the required dependencies
pip install .
```

## Sweeping Instructions

All experiments are running from the root directory of the repo. We use ``hydra-core`` as an ``argparse`` on steroids, in tandem with ``wandb`` for sweeping. For a sweeping configuration ``experiments/my_sweep.yaml``, run:

```
wandb sweep experiments/my_sweep.yaml
```

for which ``wandb`` responds with:

```
wandb: Creating sweep from: experiments/my_sweep.yaml
wandb: Creating sweep with ID: <my_sweep_id>
wandb: View sweep at: https://wandb.ai/<my_wandb_team>/<my_project>/sweeps/<my_sweep_id>
```

### Sweeping locally
Run sweep agent with: ``wandb agent <my_wandb_team>/<my_project>/<my_sweep_id>``.

### Sweeping with ``slurm``

``sbatch`` files format pre-configured for the Vaughan cluster. Edit the templates at will. 

We execute a script to replace the ``wandb agent ...`` line in our ``.slrm`` files:

```
./experiments/replace_wandb_agent.sh "wandb agent <my_wandb_team>/<my_project>/<my_sweep_id>"
```

Finally, spam jobs on the cluster and maximize your allocation per ``qos``: 

```
./experiments/sbatch_all.sh
```

Edit this script per your allocation. 


## Usage and Tutorials

Coming soon.

## Citation

Coming soon.



