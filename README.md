# Investigating MLP

```yml
Project Status: Functional for Input Layer only, with shoddy UX
```

The Multi-Layered Perceptron (MLP), is the simplest type of neural network, at
least conceptually. One of its quirky properties is that the indefinite article 
you use for it in English depends on whether you abbreviate it or not. It is 
also known as the Dense Neural Network (DNN), the Fully Connected Network (FCN),
and so on.

Inside an MLP are multiple "layers". These layers contain weight matrices and
bias vectors, and these in turn have associated gradients. As we train an MLP,
the values of the elements of these matrices and vectors, as well as the values
of their gradients, all change with every iterative step of the optimizer.

In this project, I am going to exploit the interactive properties of HTML files
to build an interactive tool to understand how the values and gradients of the 
weights and biases change in an MLP with each optimization step during training.

The MLP used here is a simple model with 4D input, 3D output and hidden layers 
of size 7D and 5D. It is trained on the Iris dataset from Scikit Learn, since
it is simple and elegant to train on. I am not going to spend my sparse brain
cells on complicated NN stuff here, I have a UI to build and an MLP to 
investigate.

Section 1 below enumerates the steps involved in running the code. Section 2 
states the objectives of this projects and marks the objectives which are 
accomplished. 

### Analysis Tool

There is a tool here to analyze the dataset if you would so wish. It is called
`analysis.py`, because I am highly original. 

You need to invoke it with two command line arguments, which are both integers 
in the set {0, 1, 2, 3}. They are the indices of two of the four features in the
dataset, which will be plotted on the X and Y axes respectively.

You are viewing the Iris dataset of Scikit-Learn. It is a public dataset, and 
details are available 
[here](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).

## 1. Running the Code

0. Install Python 3, if you do not have it on your system. Also install `pip`
1. Install the required packages using `requirements.txt`
2. Run the script `main.py` using Python
3. Open the HTML file `investigate.html` and control the interactive bar on top

## 2. Objectives

```
[X] Build the analysis tool for some quick EDA (and do the EDA)
[X] Build the data loader (loading.py)
[X] Build the model (model.py)

[X] Build a make-shift version using Matplotlib saves as Proof of Concept
    [X] Build the main training code (main.py)
    [X] Build the HTML tool for investigating (investigate.html)

[ ] Build the actual software
    [.] Build the main training code (main.py)
    [.] Build the HTML tool for investigating (investigate.html)
    [.] Build the relevant Javascript files for the HTML tool
    [.] Build the relevant CSS for the HTML tool

* The dot above means it has only been implemented for Input Layer
```