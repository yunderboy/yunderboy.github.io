---
title:  "Part 1 - A brain in 5 minutes"
date:   2018-06-28 17:30:44 +0200
categories: introduction-to-neural-networks
---
![](https://cdn-images-1.medium.com/max/1000/1*yB4lM-EZHhNpXiktU42n-g.jpeg)

This is my virgin blog post, so please be gentle.

The purpose of this post is to give you the introduction to artifical neural networks (ANNs) that I wish I’d got.

This is the first post in what will hopefully become a small series. I currently plan to release posts describing the following topics:

A feed forward ANN from scratch (this post)
Gradient descend and back propagation
A feed forward ANN on the GPU
Biological optimization/evolutionary AI
Reinforcement learning in games
Some psychological and philosophical posts.

# Writing a neural network from scratch

In essence, most machine learning algorithms work by having a massive amounts of knobs/variables that are somehow added together. By continually tweaking the values of these knobs, you can start to generate a network that can generalize on the features of it’s input.

Artificial neural networks is a very popular implementation of such an algorithm.

In this post i will show you how to implement a “fully connected, feed forward neural network” or simply “a multilayer perceptron” which is the most traditional type of neural network.

In practice this means that we create a network of “neurons” that are connected by “synapses” (the knobs in this kind of algorithm), the network i structured in the following way:
1. The input data only flows one way, forward (hence the “feed forward” part)
2. That each neuron in one layer, has connections to each neuron i the next layer (hence “fully connected”)

The purpose of this network is to perform binary classification, i.e figure out whether the features that is fed into the network, describes one thing or another.

The network will use the data from the table below to figure out where the output should be 0 or 1:
![](https://cdn-images-1.medium.com/max/1000/1*8HBlbOFzAEPari9_M6jF9Q.png)

The neural network works by computing “the summed weights” of a neuron. In practice this means that for each neuron in the hidden layer the values of the weights connected to that neuron is multiplied by the value of the neuron in the input layer.

An “activation function”, written as Ø(x), is then applied to the weighted sum of the input neurons and you now have the value of the neurons in the hidden layer.
![](https://cdn-images-1.medium.com/max/1000/1*M0vBOSWrtJZaqSwL87jVfg.jpeg)

The exact same operation is then performed for the output layer, each weight and it’s corresponding neuron in the hidden layer is multiplied, and the sum acts as the input for the activation function.

A neural network can use a lot of different activation functions, which one you pick depends on the type of problem that you’re trying to solve.

In this example I’m using the historically, most popular one; the Sigmoid function that has the following notation:
![](https://cdn-images-1.medium.com/max/1000/1*YtnVJ-lQ5h8ceZvlSCYvvw.png)

And the following graph:
![](https://cdn-images-1.medium.com/max/1000/1*TZIqiVX__LhpTr9C34Su9w.png)

This is all implemented in code through matrix multiplication, as you will see, the entire network can be represented as matrices.

And now for the exciting part, the actual coding:

{% highlight python %}
# Import the required libraries
import numpy as np
{% endhighlight %}

This tutorial only utilized numpy. Next we’ll set a seed for numpy, such that the otherwise random values will become deterministic, which allows for reproducibility, and thus testability:

{% highlight python %}
np.random.seed(1)
{% endhighlight %}

The training data is then initialized as a 4x3 matrix and a 4x1 matrix

{% highlight python %}
# Initialize the training set
training_set_inputs = np.array(
  [
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
  ]
)
training_set_outputs = np.array(
  [[0, 1, 1, 0]]
).T
{% endhighlight %}

The layers of the ANN; the data has 3 features and is supposed to perform binary classification, thus it has a single output. I’ve, somewhat arbitrarily, decided to got with a hidden layer of 2 neurons.

Thus we end up with the following layout:
{% highlight python %}
  layers = (
      # The amount of neurons in the 1. layer (the input layer)
      3,
      # The amount of neurons in the 2. layer (the hidden layer)
      2,
      # The amount of neurons in the 3. layer (the output layer)
      1
  )
{% endhighlight %}

Next up we define the Sigmoid activation function:
{% highlight python %}
  def sigmoid(x):
      return 1 / (1 + np.exp(-x))
{% endhighlight %}

Then we initialize the weights as matrices (this will be explained) of the following dimensions:

3x2 (amount of neurons in input layer times the amount of neurons in the hidden layer)
2x1 (amount of neurons in the hidden layer times the amount of neurons in the output layer)

{% highlight python %}
  # Initialize the weights between the input layer and the hidden layers
  w_1 = 2 * np.random.random([layers[0], layers[1]]) -1
  # Initialize the wieghts between the hidden layer and the output layer
  w_2 = 2 * np.random.random([layers[1], layers[2]]) -1
{% endhighlight %}


The range of values in the weight matrices ranges from -1 to 1.

The output will end up looking like this:

{% highlight python %}
  Dimensions of w_1: (3, 2)

  The w_1 matrix
   [[-0.1783773  -0.19704083]
   [-0.36523211  0.24383874]
   [-0.13950546  0.94760416]]
  Dimensions of w_2: (2, 1)

  The w_2 matrix
   [[ 0.35560178]
   [-0.60286022]]
{% endhighlight %}

Finally we’ll create the “predict” function that is responsible for taking a 3x1 matrix as input.

{% highlight python %}
  def predict(x):
      # Compute the values of the neurons in the hidden layer
      hidden_layer = sigmoid(np.dot(x, w_1))

      # Compute the output value
      output_layer = sigmoid(np.dot(hidden_layer, w_2)[0])

      # Binarization of the output neuron
      if output_layer <= 0:
          return 0
      if output_layer > 0:
          return 1
  # predict(np.array([0, 0, 1]))
  # returns 1
{% endhighlight %}

Allright, so a few things;

And why is the output 1 when it should be zero?
How come this sort of work (no error is spewed)
To answer your first question; it gives you the wrong output because we haven’t trained the network or “tweaked the knobs” yet. I will be doing this in the next post using a technique called “stochastic gradient descend” which is a way to figure out how much you should tweak each knob without going through all possible combinations.

In order to answer your second question I will briefly remind you of the weighted sums algorithm:
![](https://cdn-images-1.medium.com/max/1000/1*te8hFNdxVNccU4lEEQHZPg.png)

I’m aware that this might look a bit intimidating, but I’ll try to break it down as it, as math is, quite logical:

The Zm part is an alias for each neuron in the next layer of the network.

The Ø is the activation function, which in this posts case, is the sigmoid function. The sigma symbol means is essentially a for loop in the math world, it performs some sort of calculation, and sums the result of all these calculations. This one sums the product of an input neurons value (1 or 0 in this case) and the value of the weight that connects this neuron to the Z neuron that you’re currently attempting to compute the value for.

In pseudo code:
{% highlight python %}
for each neuron in the next layer
for each weight connected to this neuron
compute the product of the weight times the value of the neuron             connected to this weight in the previous layer
Sum/add all the computed products together and put the result into the sigmoid function.
{% endhighlight %}

I hope this explanation of the weighted sums algorithm made some sense.

The final outstanding question is; how was this implemented in the code?

The answer to this is, through linear algebra magic, specifically matrix multiplication and “broadcasting”.

If you’ve got no clue what linear algebra is, it’s essentially the discipline of math that deals with vectors and matrices. A channel called 3blue1brown has an incredible series on the the topic: essence of linear algebra.

So we actually compute the weigthed sums through matrix multiplication.

Let’s take neuron Z1 (a neuron in the hidden layer) as an example (refer to fig. 1.). Z1 has 3 weights connected to it, let’s call them W1, W2 and W3.

These weights are actually represented in the w_1 variable that we created in the code. w_1 looks like this:
{% highlight python %}
  w_1 =[
   [-0.1783773  -0.19704083]
   [-0.36523211  0.24383874]
   [-0.13950546  0.94760416]
  ]
{% endhighlight %}

This is actually a matrix, or a 2 dimensional list. Each row in the matrix actually represents a neuron in the input layer, and each column, a neuron in the hidden layer.

So for the sake of the example, we could say that all values in the 1. column represented weights connected to the Z1 neuron, and the values in the 2. column, the Z2 neuron.

The values of the input neurons could be this:
{% highlight python %}
  input = [0, 0, 1]
{% endhighlight %}

Let’s say that the first row of the matrix represents X1, the second X2 and the third (you guessed it) X3.

Thus we would end up computing the values of Z1 and Z2 in the following way:

Z1 = Ø(0 * -0.1783773 + 0 * -0.36523211 + 1 * -0.13950546) = 0.46518009

and Z2 = Ø(0 * -0.19704083 + 0 * 0.24383874 + 1 * 0.94760416) = 0.7206331

Written in (pseudo) mathematical notation
{% highlight python %}
  [0]    [  -0.1783773   0 * -0.19704083 ]
  [0]  * [  -0.36523211  0 * 0.24383874  ]  =
  [1]    [  -0.13950546  1 * 0.94760416  ]
  [ 0 * -0.1783773   0 * -0.19704083 ]
  [ 0 * -0.36523211  0 * 0.24383874  ] =
  [ 1 * -0.13950546  1 * 0.94760416  ]
  [  0            0          ]
  [  0            0          ]
  [  -0.13950546  0.94760416 ]
{% endhighlight %}

The “broadcasting” technique is used in cases where you’re attempting to multiply 2 matrices with different dimensions, in this case 3x1 and 3x2.

This is not allowed, therefore numpy does a bit of magic, it simply “stretches”the matrix with the smaller dimensions, so that the input matrix actually look like this:
{% highlight python %}
[0, 0]
[0, 0]
[1, 1]
{% endhighlight %}
Et voila, you now have 2 3x2 matrices that can be multiplied.

## Conclusion
So you’ve might by now be able to see that you can represent both neurons and the weights as vectors and matrices. You saw how to compute the weighted sum of each neuron through matrix multiplication and broadcasting.

Next up I will be implementing the part of the code that will be able to actually train the ANN, so stay tuned!

The code for this ANN is also available as a jupyter notebook on git.
