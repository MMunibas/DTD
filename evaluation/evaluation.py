import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

# whether an accuracy evaluation should be done, i.e., if RMSD and R2 values should be calculated
calculate_accuracy_measures = True
if calculate_accuracy_measures:
    RMSD_list_Etrans_prod = []
    RMSD_list_v_prod = []
    RMSD_list_j_prod = []
    R2_list_Etrans_prod = []
    R2_list_v_prod = []
    R2_list_j_prod = []

# specify what type of accuracy should be evaluated (NN or QCT)
# accuracy_type = 'NN'
accuracy_type = 'NN'

# whether to plot the data and the corresponding predictions
plotting = True
if plotting:
    pdf = matplotlib.backends.backend_pdf.PdfPages("predictions.pdf")
    plt.rcParams.update({'figure.figsize': [5, 3.5]})
    plt.rcParams.update({'legend.frameon': False})

# define function which finds returns the index of the element in an array closest to a specific value


def find_nearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index

# define activation function for the hidden layers


def shiftedsoftplus(x):
    return np.log(np.exp(x)+1)-np.log(2)


# define functions for the RKHS method


# gaussian reproducing kernel
def gauss_ker(x, xi, sigma):
    return np.exp(-abs(x-xi)**2/(2*sigma**2))

# function which given the kernel coefficients, the hyperparameters sigma and the grid returns the value of the RKHS-based approximation at given points (ker_grid)


def ker_eval(ker_coeff, grid, ker_grid, sigma):
    p_ker = np.zeros((len(ker_grid), 1))

    for j in range(len(ker_grid)):
        for i in range(len(grid)):
            p_ker[j, 0] += ker_coeff[i]*gauss_ker(ker_grid[j], grid[i], sigma[i])
    return p_ker

# function which returns the inverted kernel (K) matrix given the grid as well as the hyperparameters lmbda (regularization rate) and sigma


def Kinv_matrix(grid, lmbda, sigma):
    K = np.zeros((len(grid), len(grid)))

    for i in range(len(grid)):
        for j in range(len(grid)):
            if i == j:
                K[i, j] = gauss_ker(grid[j], grid[i], sigma[j]) + lmbda
            else:
                K[i, j] = gauss_ker(grid[j], grid[i], sigma[j])
    return np.linalg.inv(K)

# function which returns the value for the hyperparameter sigma at each grid points as the larger of the two neighbouring grid spacings


def get_sigma_1(grid):
    sigma = []
    for i in range(len(grid)):
        if i > 0 and i < len(grid)-1:
            left_length = (grid[i]-grid[i-1])
            right_length = (grid[i+1]-grid[i])

            if left_length > right_length:
                sigma.append(left_length)
            else:
                sigma.append(right_length)

        # for the first and last grid point sigma is assigned the value of the right/left grid spacing
        elif i == 0:
            sigma.append((grid[1]-grid[0]))
        elif i == len(grid)-1:
            sigma.append((grid[len(grid)-1]-grid[len(grid)-2]))
    return np.array(sigma)

# function which returns the value for the hyperparameter sigma at each grid points as the average neighbouring grid spacing


def get_sigma_2(grid):
    distances = []
    for i in range(len(grid)-1):
        distances.append(grid[i+1]-grid[i])
    sigma = np.mean(distances)
    return np.full(grid.shape, sigma)


# set the regularization rate lambda for the reactant and product state distributions
lambda_Etrans_reac = 0.015
lambda_v_reac = 0.012
lambda_j_reac = 0.01
lambda_Etrans_prod = 0.025
lambda_v_prod = 0.005
lambda_j_prod = 0.01

# define the grids for the reactant and product state distributions
grid_Etrans_reac = np.array([0, 0.2, 0.4, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0,
                             3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5])
grid_Etrans_prod = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5,
                             5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
grid_v_reac = np.array([0, 2, 4, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36])
grid_v_prod = np.array([0, 2, 4, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 42, 47])
grid_j_reac = np.array([0, 15, 30, 45, 60, 90, 120, 150, 180, 210, 240])
grid_j_prod = np.array([0, 20, 40, 60, 80, 100, 125, 150, 175, 200, 220, 240])

# calculate the number of features and outputs based on these grids
num_inputs = len(grid_Etrans_reac) + len(grid_v_reac) + len(grid_j_reac)
num_outputs = len(grid_Etrans_prod) + len(grid_v_prod) + len(grid_j_prod)
print('Number of NN inputs: ' + str(num_inputs))
print('Number of NN outputs: ' + str(num_outputs))

# set the hyperparameter sigma for the gaussian reproducing kernels for the product state distrubtions
sigma_Etrans_prod = get_sigma_1(grid_Etrans_prod)
sigma_v_prod = get_sigma_2(grid_v_prod)
sigma_j_prod = get_sigma_1(grid_j_prod)

# calculate the inverse kernel matrix for the product state distributions
Kinv_Etrans_prod = Kinv_matrix(grid_Etrans_prod, lambda_Etrans_prod, sigma_Etrans_prod)
Kinv_v_prod = Kinv_matrix(grid_v_prod, lambda_v_prod, sigma_v_prod)
Kinv_j_prod = Kinv_matrix(grid_j_prod, lambda_j_prod, sigma_j_prod)


# import data indices and the corresponding temperatures
data = np.genfromtxt('../data_preprocessing/input_and_reference_all.txt', delimiter=',')
test_indices = np.genfromtxt('../data_preprocessing/test_indices.txt').astype(int)
temperatures_all = np.genfromtxt('../data/tinput.dat', delimiter='  ')

# import weights and biases of the trained NN
h0W = np.loadtxt('../training/NN_parameters/Coeff_h0W.dat', delimiter=',', unpack=True)
h0b = np.loadtxt('../training/NN_parameters/Coeff_h0b.dat', delimiter=',', unpack=True)
h1W = np.loadtxt('../training/NN_parameters/Coeff_h1W.dat', delimiter=',', unpack=True)
h1b = np.loadtxt('../training/NN_parameters/Coeff_h1b.dat', delimiter=',', unpack=True)
outW = np.loadtxt('../training/NN_parameters/Coeff_outW.dat', delimiter=',', unpack=True)
outb = np.loadtxt('../training/NN_parameters/Coeff_outb.dat', delimiter=',', unpack=True)

# import mean values and standard deviations for standardization
mval_input = np.loadtxt('../training/NN_parameters/Coeff_mval_input.txt',
                        delimiter=',', unpack=True)
stdv_input = np.loadtxt('../training/NN_parameters/Coeff_stdv_input.txt',
                        delimiter=',', unpack=True)
mval_output = np.loadtxt('../training/NN_parameters/Coeff_mval_output.txt',
                         delimiter=',', unpack=True)
stdv_output = np.loadtxt('../training/NN_parameters/Coeff_stdv_output.txt',
                         delimiter=',', unpack=True)

# loop through all data sets
inputs = np.zeros(num_inputs)
outputs = np.zeros(num_outputs)
aout = np.zeros(num_outputs)
type = ''
for i in test_indices:
    print('Current data set: ' + str(i))
    temperatures = temperatures_all[i-1]

    # calculate the NN predictions for a given dataset

    # standardize the features and output from the dataset
    for j in range(num_inputs):
        inputs[j] = (data[i-1][j] - mval_input[j]) / stdv_input[j]
    for k in range(num_inputs, num_inputs + num_outputs):
        outputs[k-num_inputs] = data[i-1][k]

    # extract the reference amplitudes
    p_ref_Etrans_prod = outputs[:len(grid_Etrans_prod)]
    p_ref_v_prod = outputs[len(grid_Etrans_prod):len(grid_v_prod)+len(grid_Etrans_prod)]
    p_ref_j_prod = outputs[len(grid_v_prod)+len(grid_Etrans_prod):num_outputs]

    # calculate the predicted amplitudes
    a0 = shiftedsoftplus((np.matmul(h0W, inputs) + h0b))
    a1 = shiftedsoftplus((np.matmul(h1W, a0) + h1b))

    aout_standardized = 8.0*np.tanh(np.matmul(outW, a1) + outb)
    aout = aout_standardized * stdv_output + mval_output

    p_pred_Etrans_prod = aout[:len(grid_Etrans_prod)]
    p_pred_v_prod = aout[len(grid_Etrans_prod):len(grid_v_prod)+len(grid_Etrans_prod)]
    p_pred_j_prod = aout[len(grid_v_prod)+len(grid_Etrans_prod):num_outputs]

    ################################################################

    # Go through the product state distributions, calculate the continuous predictions based on the RKHS method and finally evaluate the prediction accuracy and construct plots if specified

    # product relative translational energy distributions

    # load the product relative translational energy distribution data
    x, p = np.loadtxt('../data/pe' + str(i) + '.dat', unpack=True)

    # assign variables
    p_pred = p_pred_Etrans_prod
    p_ref = p_ref_Etrans_prod
    grid = grid_Etrans_prod
    sigma = sigma_Etrans_prod
    Kinv = Kinv_Etrans_prod

    # define the grid on which to evaluate the RKHS-based approximation for plotting
    ker_grid = np.linspace(0.0, 30.0, 300)

    # calculate the kernel coefficients from the predicted amplitudes and evaluate the RKHS-based approximation at the chosen set of points
    ker_coeff_pred = np.matmul(Kinv, p_pred)
    p_ker_pred = ker_eval(ker_coeff_pred, grid, ker_grid, sigma)

    # calculate the kernel coefficients from the reference amplitudes and evaluate the RKHS-based approximation at the chosen set of points
    ker_coeff_ref = np.matmul(Kinv, p_ref)
    p_ker_ref = ker_eval(ker_coeff_ref, grid, ker_grid, sigma)

    # do an accuracy evaluation if specified, i.e., calculate the RMSD and R2 values based on the current distribution (relative translational energy)
    if calculate_accuracy_measures:

        # evaluate the RKHS-based approximation constructed from the predicted amplitudes at the points where QCT data is available and divide by the normalization factor obtained by numerical integration of the distributions obtained by QCT simulations
        p_pred_acc = ker_eval(ker_coeff_pred, grid, x, sigma)/np.trapz(p, x)

        # if RMSD_QCT and R2_QCT should be calculated
        if accuracy_type == 'QCT':

            # calculate the locally averaged distribution from the QCT data with n_max = 3
            p_ref_acc = []
            for j in range(len(p)):
                if j-1 < 0 or j+1 > (len(p)-1):
                    p_ref_acc.append(p[j])
                elif j-2 < 0 or j+2 > (len(p)-2):
                    p_ref_acc.append(np.mean(p[j-1:j+1+1]))
                elif j-3 < 0 or j+3 > (len(p)-3):
                    p_ref_acc.append(np.mean(p[j-2:j+2+1]))
                else:
                    p_ref_acc.append(np.mean(p[j-3:j+3+1]))

            # divide by the normalization factor obtained by numerical integration of the distributions obtained by QCT simulations
            p_ref_acc = p_ref_acc/np.trapz(p, x)

        # if RMSD_NN and R2_NN should be calculated
        elif accuracy_type == 'NN':

            # evaluate the RKHS-based approximation obtained by the reference amplitudes at the points where QCT data is available and divide by the normalization factor obtained by numerical integration of the distributions obtained by QCT simulations
            p_ref_acc = ker_eval(ker_coeff_ref, grid, x, sigma)/np.trapz(p, x)

        # calculate the corresponding RMSD value
        sum = 0
        for j in range(len(p_pred_acc)):
            if p_pred_acc[j] < 0:
                p_pred_acc[j] = 0.0
            if p_pred_acc[j] != p_ref_acc[j]:
                sum = sum + (p_pred_acc[j]-p_ref_acc[j])**2

        RMSD = float(np.sqrt(sum/len(p_pred_acc)))
        RMSD_list_Etrans_prod.append(RMSD)

        # calculate the corresponding R2 value
        RSS = sum
        SStot = np.sum((p_ref_acc-np.mean(p_ref_acc))**2)
        R2 = float(1 - RSS/SStot)
        R2_list_Etrans_prod.append(R2)

    # plot the distributions obtained by QCT simulations as well as the corresponding model predictions and predicted amplitudes
    if plotting:
        plt.figure()
        plt.plot(x, p, '-k', label='QCT')
        plt.plot(grid, p_pred, '.r', label='G-NN')
        plt.plot(ker_grid, p_ker_pred, '-r')
        plt.plot(grid, p_ref, '.g', label='G-R')
        plt.plot(ker_grid, p_ker_ref, '-g')
        plt.xlabel(r"$E_{\mathrm{trans}}'$" + ' [eV]')
        plt.ylabel('Probability')
        plt.title(r"$P(E_{\mathrm{trans}}')$")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # --------------------------------------------------------------------------

    # similarly for the product relative translational energy distributions

    x, p = np.loadtxt('../data/pv' + str(i) + '.dat', unpack=True)

    p_pred = p_pred_v_prod
    p_ref = p_ref_v_prod
    grid = grid_v_prod
    sigma = sigma_v_prod
    Kinv = Kinv_v_prod

    ker_grid = np.linspace(0.0, 70.0, 300)
    ker_coeff_pred = np.matmul(Kinv, p_pred)
    p_ker_pred = ker_eval(ker_coeff_pred, grid, ker_grid, sigma)

    ker_coeff_ref = np.matmul(Kinv, p_ref)
    p_ker_ref = ker_eval(ker_coeff_ref, grid, ker_grid, sigma)

    if calculate_accuracy_measures:
        p_pred_acc = ker_eval(ker_coeff_pred, grid, x, sigma)/np.sum(p)

        if accuracy_type == 'QCT':

            # n_max = 2
            p_ref_acc = []
            for j in range(len(p)):
                if j-1 < 0 or j+1 > (len(p)-1):
                    p_ref_acc.append(p[j])
                elif j-2 < 0 or j+2 > (len(p)-2):
                    p_ref_acc.append(np.mean(p[j-1:j+1+1]))
                else:
                    p_ref_acc.append(np.mean(p[j-2:j+2+1]))

            p_ref_acc = p_ref_acc/np.sum(p)

        elif accuracy_type == 'NN':
            p_ref_acc = ker_eval(ker_coeff_ref, grid, x, sigma)/np.sum(p)

        sum = 0
        for j in range(len(p_pred_acc)):
            if p_pred_acc[j] < 0:
                p_pred_acc[j] = 0.0
            if p_pred_acc[j] != p_ref_acc[j]:
                sum = sum + (p_pred_acc[j]-p_ref_acc[j])**2

        RMSD = float(np.sqrt(sum/len(p_pred_acc)))
        RMSD_list_v_prod.append(RMSD)

        RSS = sum
        SStot = np.sum((p_ref_acc-np.mean(p_ref_acc))**2)
        R2 = float(1 - RSS/SStot)
        R2_list_v_prod.append(R2)

    if plotting:
        plt.figure()
        plt.plot(x, p, '-k', label='QCT')
        plt.plot(grid, p_pred, '.r', label='G-NN')
        plt.plot(ker_grid, p_ker_pred, '-r')
        plt.plot(grid, p_ref, '.g', label='G-R')
        plt.plot(ker_grid, p_ker_ref, '-g')
        plt.xlabel(r"$\it{v'}$")
        plt.ylabel('Probability')
        plt.title(r"$P(\it{v'})$")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # --------------------------------------------------------------------------

    # similarly for the product relative translational energy distributions

    x, p = np.loadtxt('../data/pj' + str(i) + '.dat', unpack=True)

    p_pred = p_pred_j_prod
    p_ref = p_ref_j_prod
    grid = grid_j_prod
    sigma = sigma_j_prod
    Kinv = Kinv_j_prod

    ker_grid = np.linspace(0.0, 320.0, 600)
    ker_coeff_pred = np.matmul(Kinv, p_pred)
    p_ker_pred = ker_eval(ker_coeff_pred, grid, ker_grid, sigma)

    ker_coeff_ref = np.matmul(Kinv, p_ref)
    p_ker_ref = ker_eval(ker_coeff_ref, grid, ker_grid, sigma)

    if calculate_accuracy_measures:
        p_pred_acc = ker_eval(ker_coeff_pred, grid, x, sigma)/np.sum(p)

        if accuracy_type == 'QCT':

            # n_max = 10
            p_ref_acc = []
            for j in range(len(p)):
                if j-1 < 0 or j+1 > (len(p)-1):
                    p_ref_acc.append(p[j])
                elif j-2 < 0 or j+2 > (len(p)-2):
                    p_ref_acc.append(np.mean(p[j-1:j+1+1]))
                elif j-3 < 0 or j+3 > (len(p)-3):
                    p_ref_acc.append(np.mean(p[j-2:j+2+1]))
                elif j-4 < 0 or j+4 > (len(p)-4):
                    p_ref_acc.append(np.mean(p[j-3:j+3+1]))
                elif j-5 < 0 or j+5 > (len(p)-5):
                    p_ref_acc.append(np.mean(p[j-4:j+4+1]))
                elif j-6 < 0 or j+6 > (len(p)-6):
                    p_ref_acc.append(np.mean(p[j-5:j+5+1]))
                elif j-7 < 0 or j+7 > (len(p)-7):
                    p_ref_acc.append(np.mean(p[j-6:j+6+1]))
                elif j-8 < 0 or j+8 > (len(p)-8):
                    p_ref_acc.append(np.mean(p[j-7:j+7+1]))
                elif j-9 < 0 or j+9 > (len(p)-9):
                    p_ref_acc.append(np.mean(p[j-8:j+8+1]))
                elif j-10 < 0 or j+10 > (len(p)-10):
                    p_ref_acc.append(np.mean(p[j-9:j+9+1]))
                else:
                    p_ref_acc.append(np.mean(p[j-10:j+10+1]))

            p_ref_acc = p_ref_acc/np.sum(p)

        if accuracy_type == 'NN':
            p_ref_acc = ker_eval(ker_coeff_ref, grid, x, sigma)/np.sum(p)

        sum = 0
        for j in range(len(p_pred_acc)):
            if p_pred_acc[j] < 0:
                p_pred_acc[j] = 0.0
            if p_pred_acc[j] != p_ref_acc[j]:
                sum = sum + (p_pred_acc[j]-p_ref_acc[j])**2

        RMSD = float(np.sqrt(sum/len(p_pred_acc)))
        RMSD_list_j_prod.append(RMSD)

        RSS = sum
        SStot = np.sum((p_ref_acc-np.mean(p_ref_acc))**2)
        R2 = float(1 - RSS/SStot)
        R2_list_j_prod.append(R2)

    if plotting:
        plt.figure()
        plt.plot(x, p, '-k', label='QCT')
        plt.plot(grid, p_pred, '.r', label='G-NN')
        plt.plot(ker_grid, p_ker_pred, '-r')
        plt.plot(grid, p_ref, '.g', label='G-R')
        plt.plot(ker_grid, p_ker_ref, '-g')
        plt.xlabel(r"$\it{j'}$")
        plt.ylabel('Probability')
        plt.title(r"$P(\it{j'})$")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

# close all pdf files if data and predictions were plotted
if plotting:
    pdf.close()
    plt.close('all')

# calculate the overall accuracy measures
if calculate_accuracy_measures:

    # calculate the overall RMSD value through averaging over all distributions of all data sets
    RMSD_Etrans_prod = np.mean(RMSD_list_Etrans_prod)
    RMSD_v_prod = np.mean(RMSD_list_v_prod)
    RMSD_j_prod = np.mean(RMSD_list_j_prod)
    RMSD_overall = 1/3*(RMSD_Etrans_prod + RMSD_v_prod + RMSD_j_prod)

    # calculate the overall R2 value through averaging over all distributions of all data sets
    R2_Etrans_prod = np.mean(R2_list_Etrans_prod)
    R2_v_prod = np.mean(R2_list_v_prod)
    R2_j_prod = np.mean(R2_list_j_prod)
    R2_overall = 1/3*(R2_Etrans_prod + R2_v_prod + R2_j_prod)

    # save the results in a txt file
    with open("./prediction_accuracy_{}.txt".format(accuracy_type), "w") as txt_file:
        txt_file.write('RMSD_overall: ' + str(RMSD_overall) + ', R2_overall: ' + str(R2_overall))
