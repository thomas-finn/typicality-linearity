# Searching for a global typicality direction

The purpose of this suite of experiments is to use the Rosch dataset and corresponding prompts to find a typicality direction both per-category and across categories. 

Experiments will be performed on `Qwen2.5-1.5B` first for preliminary results and then `Qwen2.5-7B` to see if scaling the model up has any meaningful differences. Later we will experiment to see if the observed behaviour extends to other model classes.

## Preliminaries to experiments

### Extracting activations

First we extract the activations at layers 10, 15, 20, and 25 on the end-of-sentence period token on each of the prompts constructed from the Rosch dataset.
This will allow us to see of internal representations change between early-middle and late-middle layers where abstract representations are typically created in LLMs.

There will be four types of activations that will be used for experiments:
 - Raw activations with extra transformations
 - Adding a mean category contrast subtraction where
$$
    a(i,c,p,\ell) - \mathbb{E}_{c'\neq c}\left[a(i,c',p, \ell)\right]
$$
where $a(i,c,p,\ell)$ is the activation for the item $i$, category $c$, prompt format $p$, and at layer $\ell$.
 - Adding a mean item contrast subtraction where
$$
    a(i,c,p,\ell) - \mathbb{E}_{i'\neq i}\left[a(i',c,p, \ell)\right].
$$

We will explore other concept-erasure methods in our experiments but for the mean contrast methods we will need to extract activations beforehand. 
For the category contrast method we will use all the other categories in the Rosch data.
For the item contrast we will select nine random items that are not a member of that category. 
Some items do appear in multiple categories so some care will be needed here.

### PCA plots

Just before we begin our probe experiments, we will visualise how typicality is represented in the first few principal components of the residual stream. 

In `experiments/probes/global-direction/pca` we will have a file `qwen2-5-1-5B-rosch-pca.html` that outputs visualisations for the four activation types outlined in `Extracting activations` in a 2x2 grid with typicality graded by the rating z-score and a dropdown menu that allows you to alter
 - The layer the activations where extracted from
 - The choice of category (as well as all categories)
 - The choice of prompt format
 - How many principal components (2 or three visualised)

On a second page of the html have another 2x2 grid that is a plot that describes how many dimensions the choices require for 95% of the variance in the activations to be explained to see the effective dimensionality of the different categories over prompt formats and layers. The choices from the dropdown should be as from the previous page.

## Probing experiments

### Probing per category and prompts

With the activations extracted in the prelimary work, we will now build linear probes to see how well typicality is linearly encoded in the residual stream at intermediate layers.

The output file is `qwen2-5-1-5B-probes-01.html` and will have a number of pages outling the behaviour of the probes.
First there is the performance of the probes when trained on a single category and how it performs on other categories as evaluation. When train and eval are the same category, use 5-fold CV:
 - Layer (10, 15, 20, 25)
 - Include or exclude ambiguous items (include all, exclude lenient exclude strict)
 - Difference-of-means between top half and bottom half typicality ratings, top third and bottom third typicality ratings, top quarter and bottom quarter typicality ratings.
 - Prompt format
 - Output metric (spearman, pearson, wilson-cox)
There will be four plots on a 2x2 grid over the four different activation sets (raw, mean category constrast, mean item contrast, both contrasts)

On a second page have the cosine similarity between the directions for each category over the following options:
 - Layer (10, 15, 20, 25)
 - Include or exclude ambiguous items (include all, exclude lenient exclude strict)
 - Difference-of-means between top half and bottom half typicality ratings, top third and bottom third typicality ratings, top quarter and bottom quarter typicality ratings.
 - Prompt format
There will be four plots on a 2x2 grid over the four different activation sets (raw, mean category constrast, mean item contrast, both contrasts)

On a third page have the performance of the probes when trained on a given prompt format and how it performs on other prompt formats as evaluation. When train and eval are the same format, use 5-fold CV:
 - Category
 - Layer (10, 15, 20, 25)
 - Include or exclude ambiguous items (include all, exclude lenient exclude strict)
 - Difference-of-means between top half and bottom half typicality ratings, top third and bottom third typicality ratings, top quarter and bottom quarter typicality ratings.
 - Prompt format
 - Output metric (spearman, pearson, wilson-cox)
There will be four plots on a 2x2 grid over the four different activation sets (raw, mean category constrast, mean item contrast, both contrasts) 

On a fourth page have the cosine similarity between the directions for each prompt over the following options:
 - Category
 - Layer (10, 15, 20, 25)
 - Include or exclude ambiguous items (include all, exclude lenient exclude strict)
 - Difference-of-means between top half and bottom half typicality ratings, top third and bottom third typicality ratings, top quarter and bottom quarter typicality ratings.
 - Prompt format
There will be four plots on a 2x2 grid over the four different activation sets (raw, mean category constrast, mean item contrast, both contrasts)

### Probing over categories and prompts

The output file for this work is `qwen2-5-1-5B-probes-02.html`.

On the first page we train probes not per category but over all categories simultaneously. The prompt formats are being trained and evaluated over. The options for the dropdown are
 - Layer (10, 15, 20, 25)
 - Include or exclude ambiguous items (include all, exclude lenient exclude strict)
 - Difference-of-means between top half and bottom half typicality ratings, top third and bottom third typicality ratings, top quarter and bottom quarter typicality ratings. 
 - Output metric (spearman, pearson, wilson-cox)
There will be four plots on a 2x2 grid over the four different activation sets (raw, mean category constrast, mean item contrast, both contrasts)

On the second page we train probes not per prompt but over all prompts simultaneously. The categories are being trained and evaluated over. The options for the dropdown are
 - Layer (10, 15, 20, 25)
 - Include or exclude ambiguous items (include all, exclude lenient exclude strict)
 - Difference-of-means between top half and bottom half typicality ratings, top third and bottom third typicality ratings, top quarter and bottom quarter typicality ratings. 
 - Output metric (spearman, pearson, wilson-cox)
There will be four plots on a 2x2 grid over the four different activation sets (raw, mean category constrast, mean item contrast, both contrasts)




