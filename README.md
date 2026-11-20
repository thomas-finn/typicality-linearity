# Concept typicality in LLMs

The purpose of this repo is to explore whether concept typicality is linearly encoded by LLMs at intermediate layers. 

Here, by concept typicality we refer to how representative an item is for a given category. 
A classic example is both robins and penguins are classified are birds, but robins are seen as a more typical member. 

**This repo represents a work in progress**

## Data

Our primary dataset is from Rosch (1975) (link to paper [here](https://psycnet.apa.org/record/1976-00172-001)) that contains item-category pairs over ten categories.
Each item-category pair was rated for how typical it is for that category by a group of humans and we take the mean human score as our typicality score.

From the item-category pairs we then create a suite of prompts varying in format and length that we use to test if concept typicality is linearly encoded at intermediate layers of an LLM.

See [`data`](./data/) for more details.

## Experiments

There are three main types of experiments that we will perform using the suite of prompts built from the Rosch data.

 - *PCA analysis* to visualise the activations.
 - *Probes* to test if typicality is linearly encoded.
 - *Steering* to test if any typicality directions discovered are causal.

Outputs from experiments and activations are stored in [`outputs`](./outputs/) but this is not committed to git due to large file sizes. This will cause some html output files to not render properly unless the required scripts are ran locally.

Activations are extracted using the scripts in [`experiments/`]

### PCA analysis

As a preliminary visual aid to our experiments, we visualise the first few principal components of activations extracted from our suite of prompts created in [`data`](./data/).

See [`experiments/pca`](./experiments/pca/) for more details. 

### Probing

Probing experiments attempt to see if simple linear probes correlate with typicality and if they do so within a given category or across all categories. 

Probing experiments will be done across layers, prompt formats, model families, model sizes, and datasets.

See [`experiments/probes`](./experiments/probes/) for more details.

### Steering

**Work in progress**

## Code

Most of the code needed to perform our experiments is in [`src/typicality_linearity`](./src/typicality_linearity/) and [`scripts/`](./scripts/) contains simple wrapper functions used to execute what is needed for experiments.

**Disclaimer**: Claude Code was used to create much of the code and I take full responsibility for it. All mistakes in this repo are mine. See [CLAUDE.md](./CLAUDE.md) and [AGENTS.md](./AGENTS.md) for more information on both executing code and how Claude has been instructed.

