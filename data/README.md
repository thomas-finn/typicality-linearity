# Data

There are two sources of data required for our experiments:
 - [`raw`](./raw/) that contains item-category pairs with some measure of typicality, usually from a study in cognitive pyschology. 
 - [`processed`](./processed/) that transforms the raw item-category data into a suite of prompts which we will extract activations on during our experiments.

For early experiments we are using the Rosch (1975) dataset containing ten categories with 50-60 items per category. See this [link](https://psycnet.apa.org/record/1976-00172-001) for the original Rosch paper. 