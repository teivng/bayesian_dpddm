# CIFAR-10.1 Experimental Notes

We are only using ``wandb`` while omitting ``hydra-core``. Core logic is in ``experiments/cifar101/exp_cifar10.py``, while ``slurm`` jobs with different ``qos`` are handled via the ``sbatch``-ing of multiple ``*.slrm`` files. 

For the next set of experiments, we might use ``hydra-core`` as an advanced ``argparse``. 