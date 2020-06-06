# DTD

Construct a distribution-to-distribution (DTD) model for a reactive atom-diatom collision system. Here, we consider a grid-based DTD model for data from Set1.

**Requirements**

python3 and TensorFlow

**Data**

The data folder contains files for 9 sets of reactant and product state distributions from Set1 that were obtained by QCT simulations. The corresponding sets of temperatures T_trans, T_rovib can be read from the tinput.dat file.

**Prepare input and reference values for training a neural network**

Run the generate_input_and_reference.py code located in the data_preprocessing folder to generate the files containing the input and reference values as well as a PDF with the corresponding plots, based on the 9 data sets from the data folder.

**Training a neural network**

The training folder already contains a file with input and reference values that were generated using the generate_input_and_reference.py code considering the complete Set1, with N_tot = 3698 and N_test = 98.

To train a neural network (NN), edit the training.py code in the training folder to specify the NN architecture and hyperparameters. Then run the training.py code by mentioning number of training data sets, validation data sets, seed, number of training epochs and batch size:

`python training.py 3000 600 11 2000 500`

**Get the optimized neural network parameters**

After finishing with the training, edit the print_coeff.py code in the training folder to load the NN model resulting in the lowest validation loss. Run the code with the same parameters as training.py to obtain the values of the corresponding optimized NN parameters in separate files:

`python print_coeff.py 3000 600 11 2000 500`

**Constructing the predicted product state distributions**

Edit the evaluation.py code in the evaluation folder to specify whether and what accuracy measures RMSD, R^2 should be calculated. Then run the code to obtain a file containing the desired accuracy measures, as well as a PDF with the corresponding plots for data sets from the data folder.

