# DTD

Construct a distribution-to-distribution (DTD) model for a reactive atom-diatom collision system. Here, we consider a grid-based DTD model for data from Set1 in reference arXiv:2005.14463.

**Requirements**

python3, TensorFlow and matplotlib

**Data**

The data folder contains files for 9 sets of reactant and product state distributions from Set1 that were obtained by quasi-classical trajectory simulations. The corresponding sets of temperatures <img src="https://render.githubusercontent.com/render/math?math=T_{trans}">, <img src="https://render.githubusercontent.com/render/math?math=T_{rovib}"> can be read from the file *tinput.dat*.

**Prepare input and reference values for training a neural network**

Run the code *generate_input_and_reference.py* located in the folder *data_preprocessing* to generate the files containing the input and reference values as well as a PDF with the corresponding plots, based on the 9 data sets from the folder *data*.

**Train a neural network**

The folder *training* already contains a file with input and reference values that were generated using the code *generate_input_and_reference.py* considering the complete Set1, with <img src="https://render.githubusercontent.com/render/math?math=N_{tot}= 3698"> and <img src="https://render.githubusercontent.com/render/math?math=N_{test}=98">.

To train a neural network (NN), edit the code *training.py* in the folder *training* to specify the NN architecture and hyperparameters. Then run the code *training.py* by mentioning number of training data sets, validation data sets, seed, number of training epochs and batch size:

`python training.py 3000 600 11 2000 500`

**Get the optimized neural network parameters**

After finishing with the training, edit the code *print_coeff.py* in the folder *training* to load the NN model resulting in the lowest validation loss. Run the code with the same parameters as *training.py* to obtain the values of the corresponding optimized NN parameters in separate files:

`python print_coeff.py 3000 600 11 2000 500`

**Construct the predicted product state distributions**

Edit the code *evaluation.py* in the folder *evaluation* to specify whether and what accuracy measures (RMSD, <img src="https://render.githubusercontent.com/render/math?math=R^{2}">) should be calculated. Then run the code to obtain a file containing the desired accuracy measures, as well as a PDF with the corresponding plots for data sets from the folder *data*.

**Cite as** Julian Arnold, Debasish Koner, Silvan Kaeser, Narendra Singh, Raymond J. Bemish, and Markus Meuwly, arXiv:2005.14463 [physics.chem-ph]
