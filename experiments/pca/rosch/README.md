# PCA analysis of Rosch data

## Qwen2.5-1.5B

### No baselines

First we consider a relatively small LLM; `Qwen2.5-1.5B`. In this experiment we consider all prompts in the Rosch suite located at `data/processed/rosch/rosch_prompts.csv`. We extract activations at two locations of the prompt at layers 10, 15, 20, and 25:
 - the final token of the item,
 - the end-of-sentence period token.

In each prompt, the item appears before the category and hence has not yet seen the category in the latter part of the sentence. Extracting at these two locations will illuminate whether typicality is encoded within an item or if typicality requires an item-category pair.

For the activations extracted, PCA analysis is performed on the top three principal components.

The output of the experiment is visualised in a html file `qwen2-5-1-5B-no-baselines.html` that has the following:
 - A dropbox to select which layer of the LLM the activations where extracted from.
 - A dropbox to select which prompt is selected from the Rosch suite.
 - A dropbox to select whether the top two or top three principcal components are visualised.
 - A dropbox to select what category is chosen.
 - Typicality is given by the hue of the points in the PCA plot, where points are large enough to make visualisation easy for the viewer.
 - Both the last token and period token plots are side by side, with the prompt appearing above each plot and highlighting which token the activations are taken from.
 - Moving over a point tells you the prompt that point refers to.

### Subtracting a mean contrast baseline

Next we run the same experiment as the no baselines example above but subtract a mean contrast baseline for each activations.
To be precise, if $a(\ell,p,i,c)$ refers to the activation at layer $\ell$ for the prompt format $p$ with item $i$ and category $c$, then rather look at the PCA plots for $a(\ell,p,i,c)$, we inspect 
$$
   a(\ell,p,i,c) - \text{mean}_{c'\neq c}\left(a(\ell,p,i,c')\right).
$$

The output should be the same format as the no baselines experiment and be stored in `qwen2-5-1-5B-mean_contrast-baseline.html`.

The HTML also shows the PCA of the same points without the baseline, side by side with the mean contrast PCA, so that the two can be compared.

