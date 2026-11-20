# Concept typicality in LLMs

The purpose of this repo is to explore whether concept typicality is linearly encoded by LLMs at intermediate layers. 

Here, by concept typicality we refer to how representative an item is for a given category. 
A classic example is both robins and penguins are classified are birds, but robins are seen as a more typical member. 

**This repo represents a work in progress**

## Data

Our primary dataset is from Rosch (1975) (link to paper here) that contains item-category pairs over ten categories.
Each item-category pair was rated for how typical it is for that category by a group of humans and we take the mean human score as our typicality score.

From the item-category pairs we then create a suite of prompts varying in format and length that we use to test if concept typicality is linearly encoded at intermediate layers of an LLM.

See [`data`](./data/) for more details.

## Experiments



### PCA analysis

As a preliminary visual aid to our experiments, we visualise the first few principal components of activations extracted from our suite of prompts created in [`data`](./data/).

See [`experiments/pca`](./experiments/pca/) for more details.

### Probing

We 

### Steering

**Work in progress**

