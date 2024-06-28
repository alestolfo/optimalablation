# Optimal ablation

Optimal ablation (OA) is a new ablation method for interpretability. OA sets a component’s value to the constant value that minimizes the expected loss incurred by the ablated model.

In this repository, we demo OA on three applications of ablation.
1. Circuit discovery ([Conmy et al., 2023](https://arxiv.org/abs/2304.14997))
2. Causal tracing ([Meng et al., 2022](https://arxiv.org/abs/2202.05262)
3. Prediction with latent representations ([Belrose et al., 2023](https://arxiv.org/abs/2303.08112))

We also propose Uniform Gradient Sampling (UGS), a methodological contribution to circuit discovery. Uniform Gradient Sampling is related to the "reparameterization trick" from [Louizos et al., 2018](https://arxiv.org/abs/1712.01312), which we refer to as HardConcrete Gradient Sampling (HCGS), but UGS samples gradients according to a mixture of the Uniform and Bernoulli distributions on ablation coefficients rather than a HardConcrete distribution.

Our experiments are built to run on GPT-2.

## Circuit discovery

### Pruners

Our code supports running HCGS and UGS. Both methods involve applying a partial ablation mask over model components with a coefficient on (0,1) for each component. To support this behavior, we use a pruner class, which is a wrapper around the base model with added hooks to perform these ablation interventions.

**pruners/VertexPruner.py** supports applying a mask of coefficients over vertices (attention heads and MLP layers) when viewing the model as a computational graph. We initialize the pruner by instantiating a VertexPruner and calling add_pruning_hooks().

**pruners/EdgePruner.py** supports applying a mask of coefficients over edges. We initialize the pruner by instantiating an EdgePruner and calling add_cache_hooks() and add_pruning_hooks(). 

**pruners/Pruner.py** is a generic base class that provides functions for both vertex and edge pruning.

### Mask samplers

The main difference between HCGS and UGS is how to sample the mask of ablation coefficients. We use a mask sampler class, which inherits from torch.nn.Module, to sample the coefficients according to some distributional parameters. When running a forward pass of each pruner class, we call _MaskSampler.forward()_ and then access the _MaskSampler.sampled_mask_ class member inside the forward pass of our pruner class to apply the ablation mask. We regularize on the distributional parameters directly by adding a loss term given by _MaskSampler.get_mask_loss()_.

**mask_samplers/MaskSampler.py** contains:
- _ConstantMaskSampler_, which allows us to supply a constant (deterministic mask). We use this class when evaluating a proposed circuit after training has converged.
- _MaskSampler_, which provides an basic implementation for HCGS on vertices.

**mask_samplers/EdgeMaskSampler.py** contains:
- _EdgeMaskJointSampler_, which extends the _MaskSampler_ class for HCGS on edges. It adds an additional vertex regularization loss term, given by _EdgeMaskSampler.node_reg_loss()_, and the functionality of pruning dangling edges.
- _EdgeMaskUnifSampler_, which we use for UGS on edges. We use a static window size by default, but this class can be modified for a dynamic window by setting _EdgeMaskUnifSampler.sampling_function = partial(EdgeMaskUnifSampler.sample_modified_unif, dynamic_window=True)_.

mask_samplers/AblationMaskSampler.py contains the SingleComponentMaskSampler class, which we use to set a constant mask in which a single component is left out.

### Other important files

**utils/MaskConfig.py** provides the MaskConfig class, which includes configuration parameters and utility functions for training and evaluating sparse circuits, including an initialization function for mask parameters, functions to save and load previous training runs, and an evaluation function for selected circuits. 

**edge_pruning_unif_dynamic.py** is the main file to run UGS on edges.

**edge_pruning_hc.py** is the main file to run HCGS on edges.

**edge_post_training.py** evaluates OA on a proposed circuit by training the ablation constants. Note that even though HCGS and UGS provide hypothesized ablation constants during training, we do not use these since we want to use the same training process to evaluate OA on circuits obtained from HCGS and UGS and those obtained with other methods (ACDC, EAP) which cannot support pre-training the ablation constants.

**edge_eval.py** evaluates all circuits in all subfolders of a given folder, which is useful because we sweep over regularization coefficients on the mask parameters to obtain circuits of different sizes.

**circuit_pareto_plot.py** plots the evaluated circuits on Delta (KL-loss) and circuit size.

### Miscellaneous files

**circuit_utils.py** contains utility functions for converting circuits between different formats (e.g. mask to list), counting circuit components, and pruning dangling edges.

**circuits_comparison.py** compares different circuits.

**circuits_random.py** generates and evaluates random circuits.

**circuits_random_plot.py** visualizes the results for random circuits.

## Causal tracing

**utils/tracing_utils.py** contains functions to run causal tracing (e.g. replacing subject token embeddings and patching in activations from clean to corrupted runs).

**causal_tracing_prep.py** contains various functions for getting the datasets in the right format, and filtering datasets for factual prompts to which the model gets the answer correct.

**casual_tracing_tokenmeans.py** computes the mean over all subject tokens (which we use as our initialization for OA).

**causal_tracing.py** trains OA for casual tracing.

**causal_tracing_eval.py** evaluates causal tracing effects with OA and Gaussian noise on a test dataset.

**causal_tracing_plot.py** generates plots.

## Prediction with latents 

We propose OCA lens, an alternative to tuned lens that maps intermediate activations at the last token position to the final-layer activation by ablating later attention layers with constants.

**utils/lens_utils.py** contains inference functions for tuned and OCA lens, and tools to evaluate the model and the lens functions with causal interventions.

**lens_tuned.py** trains tuned lens.

**lens_oa.py** trains OCA lens.

**lens_linear_oa.py** trains a linear approximation to OCA lens, which is useful for computing a linear basis for OCA lens.

**lens_compare.py** contains our main loss and causal faithfulness experiments.

**lens_eval.py** evaluates the results of these experiments.

**lens_plot.py** creates plots.

## Utility files

**utils/data.py** provides some functions to load and process [OpenWebText](https://paperswithcode.com/dataset/openwebtext), an approximation of the training dataset for GPT-2.

**utils/training_utils.py** provides general utility functions.

**utils/task_datasets.py** provides functions to load and process the Indirect Object Identification ([Wang et al., 2022](https://arxiv.org/abs/2211.00593)) and Greater-Than ([Hanna et al., 2022](https://arxiv.org/abs/2305.00586)) datasets, which are used as test cases for circuit discovery
compute_means.py computes mean activations over a dataset for use with mean ablation.

