![Boo](graphics/fantasmita.png)


# Boo: Scare-Free Gradient-boosting.

## Introduction

Boo is a library that implements tree-based gradient boosting
and part of (see below)  [extreme gradient boosting](https://github.com/dmlc/xgboost) ([reference](https://arxiv.org/abs/1603.02754)) for classification, in pure Go. 

# Features


* Simple implementation and data format. It's quite easy for any program to put the data into Boo "Databunch" format.

* The library is pure Go, so there are no runtime dependencies. There is only one compilation-time dependency (the [Gonum library](www.gonum.org)).

* The library can serialize models in JSON format, and recover them (the JSON format is pretty simple for 3rd-party libraries to read). 

* Basic file-reading  facilities a _very_ naive
reader for the libSVM format, and a reader for the CSV format), are provided.

* Cross-validation and CV-based grid search for hyperparameter optimization.



Both the regular gradient-boosting as well as the xgboost implementations are close ports/translations from the following Python implementations:

* [Gradient boosting multi-class classification from scratch](https://randomrealizations.com/posts/gradient-boosting-multi-class-classification-from-scratch/)
* [XGBoost from scratch](https://randomrealizations.com/posts/xgboost-from-scratch/)
* [Decision tree from scratch](https://randomrealizations.com/posts/decision-tree-from-scratch/)

by [Matt Bowers](https://github.com/mcb00)


## Things that are missing / in progress

Many of these reflect the fact that I mostly work with rather small, dense datasets. 

* There are only exact trees, and no sparsity-awareness.
* Several other features in the XGBoost library are absent.
* In general, computational performance is not a top priority for this project, though of course it would be nice.
* As mentioned above, the libSVM reading support is very basic. 
* Only classification is supported. Still, since its  multi-class classification using one-hot-encoding, and the "activation function" (softmax by default) can be changed, I suspect you can trick the function into doing regression by giving one class and an activation function that does nothing.
* There is nothing to deal with missing features in the samples.
* Documentation (though the tests can be used as examples, as stated below). Comments in the code are being improved.
* Ability to recover and apply serialized models from XGBoost. There is the [Leaves](https://github.com/dmitryikh/leaves) library for that, though.
* A less brute-force scheme for hyperparameter determination

On the last point, there is a preliminar, and quite naive version that uses a simple, numerical gradient-based routine to 
search for parameters.

# Using Boo

Hopefully I'll add instructions, for now, take a look at the test files (those ending in \_test.go) to see examples. The code is --mostly--commented, so the regular Go documentation system should work. (I'm working on improving the comments).

 ## On machine learning

If you want to be an informed user of Statistical/Machine learning, these are my big 3:

* [StatQuest with Josh Starmer](https://www.youtube.com/@statquest) Binge-watch worthy.
* [The Random Realizations blog](https://randomrealizations.com) Where most of the knowledge in this library comes from. Do read it.
* [An Introduction to Statistical Learning](https://www.statlearning.com/) They don't call it 'The Bible' for nothing.

(c) 2024 Raul Mera A., University of Tarapaca.

This program, including its documentation, 
is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation; either version 2.1 of the 
License, or (at your option) any later version.
          
This program and its documentation is distributed in the hope that 
it will be useful, but WITHOUT ANY WARRANTY; without even the 
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE.  See the GNU General Public License for more details.
                    
You should have received a copy of the GNU Lesser General 
Public License along with this program. If not, see 
<http://www.gnu.org/licenses/>. 

The Mascot is Copyright (c) Rocio Araya, under a [Creative Commons BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

