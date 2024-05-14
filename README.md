# LanGrader: an AI solution to evaluating arguments

**Last updated: May 14th, 2024, 1:21CST**

The project is inspired by [this paper](https://arxiv.org/pdf/1911.11408), and is a trial project prepared for Professor Vasanth Sarathy at Tufts University.

This doc (currently) serves to document the planned stages and their state of completion, but will eventually be used to document basic usage of the finished/runnable product. Ideally, this doc will be updated regularly, every 2-3 small commits or 1 large commit.

There are three stages of this project: the model, the web interface, and the extra features.

## The First Stage: The Model

In this stage, I will use a pre-trained model to create a evaluator for arguments. It should be able to provide a rating of a provided argument, with somewhat limited context. 

### Model selected:
I chose deberta-v3 from microsoft somewhat arbitrarily, as I was more or less following [this tutorial](https://huggingface.co/blog/Valerii-Knowledgator/multi-label-classification). The original paper used BERT so any model of the BERT family should probably work.

As of now, I am unsure whether have I successfully created the desired model.

### Method 1: training locally with torch/transformer

For the versions of code, please see ```p1_theModel```.

To summarize, I was unable to properly train the model. The current code uses mostly the ```transformer``` API. ```torch``` was tried earlier but to no avail.

The closest trail I got to in my 3 days of trail-and-error was for it to halt at the first validation, with the following error message:

~~~
ValueError: Mismatch in the number of predictions (12630) and references (6315)
~~~
Some efforts were made to fix this but non of the attempts worked.

TODO: 

- [ ] look for debugging methodologies; 

- [ ] add comments



### Method 2: autotrain-advanced

I then moved on to using the codeless UI provided by Huggingface,  ```autotrain-advanced```. Similar parameters were selected and I was able to train a model at about 1/2.5 of speed using method 1. The resulting model apparently have a loss of ~0.7, and could not be ran online using the online interface on hugging face.

TODO: run some test to make sure the model runs; 

## The Second Stage: Web Interface

I decided to begin the implementation of stage 1's UI to ensure some progress is made.

WIP

## The Final Stage: Extra Features

WIP


## Resources I used during this process

- stack overflow, chatGPT for debugging and general inquiry
- [Paper on argument quality ranking](https://arxiv.org/pdf/1911.11408), for inspiration and dataset
- documentation for pytorch, transformer