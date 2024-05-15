# LanGrader: an AI solution to evaluating arguments

**Last updated: May 15th, 2024, 4:16CST**

The project is inspired by [this paper](https://arxiv.org/pdf/1911.11408), and is a trial project prepared for Professor Vasanth Sarathy at Tufts University.

This doc (currently) serves to document the planned stages and their state of completion, but will eventually be used to document basic usage of the finished/runnable product. Ideally, this doc will be updated regularly, every 2-3 small commits or 1 large commit.

There are three stages of this project: the model, the web interface, and the extra features.

## The First Stage: The Model

In this stage, I will use a pre-trained model to create a evaluator for arguments. It should be able to provide a rating of a provided argument, with somewhat limited context. 

### Model selected:
I chose deberta-v3 from microsoft somewhat arbitrarily, as I was more or less following [this tutorial](https://huggingface.co/blog/Valerii-Knowledgator/multi-label-classification). The original paper used BERT so any model of the BERT family should probably work.

As of now, I am unsure whether have I successfully created the desired model.



### Some thought process

There were two approaches that came to me. The first one is rather simple: a text-regression model that spits out some value between 0 and 1 using RELU or sigmoid indicating quality. This solves the basic requirement nice and easy.

An alternative model would be more generic but expandable: multi-label classification. I got this idea from a previous project in my internship. The justification for this method is the variety of results it could provide with just a single model: it could ideally provide the topic (through generic tags like the imdb movie tags), the rating/ranking of argument (through levels 1 through 10, or some ordinal scale such as the letter grade), and the stance of the argument simultaneously. Doing so would demand more preprocessing than what the current IBM dataset provides, and most likely more data due to increased complexity.


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

Results are stored privately on my hugging face account.

TODO: run some test to make sure the model runs; 


### Method 3: Langchain.

Instead of going through the pain of training my own model, use a smaller trained LLM model such as llama3 locally, and simply feed it the data.

Two slightly different approaches were made: using the raw csv as per, and parsing it into a txt that resembles human language. 

## The Second Stage: Web Interface

I decided to begin the implementation of stage 1's UI to ensure some progress is made.

WIP

## The Final Stage: Extra Features


Some thoughts on how to realize the features

1. Detailed Feedback
    1. In practice, [Wachsmuth et al. (2017a)](https://aclanthology.org/E17-1017.pdf) would be more than enough.
    2. A simple approach would be to have a button that appears on the side of the simple results, which, upon click, automatically invokes the llm to write a short paragraph of evaluation based on each of these metrics. This might be too extensive, so an intermediate level (the three main quality outlined would suffice)
        1. Logical quality in terms of the cogency or strength of an argument.
        2. Rhetorical quality in terms of the persuasive effect of an argument or argumentation.
        3. Dialectical quality in terms of the reasonableness of argumentation for resolving issues.
    3. A switch toggle for this feature in settings might be useful
    4. From a business stand point, this seems to be a premium option due to the extensive resources needed



2. Comparison
    1. While the pair-wise comparison sounds enticing, there are two significant drawbacks: this would a) require an additional model loaded which could be challenging especially for smaller hosts b) is more or less the same when 3 or more arguments are provided
    2. Therefore, it is recommended to use the same model multiple times
    3. An idea for implementation: for the best intuitiveness:
        1. Have a toggled switch/ alternative page for multi-argument comparison
        2. Press enter after each argument; a new entry box created for each new text
            1. shift+enter creates new line within entry (no need to tell user)
        3. A click only button to submit all sentences at once
        4. the llm is asked with the intermediate response from the previous feature, then asked to select the best argument amongst all and explain why, either over all metrics or that it outstands all others significantly in one metric.



3. RefinetheUItoaccommodatenewfeaturesandensureintuitive
navigation.
    1. TODO: figma + documentation

WIP


## Resources I used during this process

- stack overflow, chatGPT for debugging and general inquiry
- [Paper on argument quality ranking](https://arxiv.org/pdf/1911.11408), for inspiration and dataset
- documentation for pytorch, transformer