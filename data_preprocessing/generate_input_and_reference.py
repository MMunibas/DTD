import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import random

# set folder_path to the folder where the data is located
folder_path = '../data/'

# whether plots should be drawn or not
plotting = True

# if so we create the corresponding pdfs to save the plots
if plotting:
    pdf = matplotlib.backends.backend_pdf.PdfPages("input_and_reference.pdf")
    plt.rcParams.update({'figure.figsize': [5, 3.5]})
    plt.rcParams.update({'legend.frameon': False})
# start_indexing and end_indexing indices labelling the data files
start_index = 1
end_index = 9
all_indices = range(start_index, end_index + 1)

# list of the prefixes with which the data files are labelled
filenames = ['re', 'rv', 'rj', 'pe', 'pv', 'pj']

# import the temperatures_alleratures associated with each data set
temperatures_all = np.genfromtxt(folder_path + 'tinput.dat', delimiter='   ')

# number of data sets that should be considered as a test set
num_test = 3

# set a seed for reproducibility
seed = 33
random.seed(seed)

# partition the data sets into training, validation, and test set
indices = list(all_indices)
test_indices = []
with open("test_indices.txt", "w") as txt_file:
    for i in range(num_test):
        index = random.choice(indices)
        indices.remove(index)
        test_indices.append(index)
        txt_file.write(str(index) + '\n')
train_valid_indices = indices

# define the grids for the reactant and product state distributions

# grid for reactant relative translational energy distributions
grid_Etrans_reac = np.array([0, 0.2, 0.4, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5,
                             7.5, 8.5, 9.5, 10.5, 11.5])

# grid for product relative translational energy distributions
grid_Etrans_prod = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5,
                             5.5, 6.5, 7.5, 8.5, 9.5, 10.5])

# grid for reactant vibrational state distributions
grid_v_reac = np.array([0, 2, 4, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36])

# grid for product vibrational state distributions
grid_v_prod = np.array([0, 2, 4, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 42, 47])

# grid for reactant rotational state distributions
grid_j_in = np.array([0, 15, 30, 45, 60, 90, 120, 150, 180, 210, 240])

# grid for product rotational state distributions
grid_j_prod = np.array([0, 20, 40, 60, 80, 100, 125, 150, 175, 200, 220, 240])

num_inputs = len(grid_Etrans_reac) + len(grid_v_reac) + len(grid_j_in)
num_outputs = len(grid_Etrans_prod) + len(grid_v_prod) + len(grid_j_prod)
print('Number of NN inputs: ' + str(num_inputs))
print('Number of NN outputs: ' + str(num_outputs))

# loop through all data sets

with open("input_and_reference_all.txt", "w") as txt_file_main:
    for i in all_indices:
        temperatures = temperatures_all[i-1]
        print('Current data set: ' + str(i))

        # for each data set loop through all reactant and product state distributions
        for filename in filenames:

            # import the distribution obtained by QCT simulations
            x, p = np.loadtxt(folder_path + filename + str(i) + '.dat', unpack=True)

            # for the reactant relative energy distributions
            if filename == 're':
                grid = grid_Etrans_reac

                # perform local averaging with n_max = 2
                p_loc_avg = []
                for j in range(len(p)):
                    if j-1 < 0 or j+1 > (len(p)-1):
                        p_loc_avg.append(p[j])
                    elif j-2 < 0 or j+2 > (len(p)-2):
                        p_loc_avg.append(np.mean(p[j-1:j+1+1]))
                    else:
                        p_loc_avg.append(np.mean(p[j-2:j+2+1]))

                # get the values of the locally averaged distributions at the grid points
                p_loc_avg_on_grid = []
                for ele in grid:
                    val = p_loc_avg[list(x).index(ele)]
                    p_loc_avg_on_grid.append(val)
                    txt_file_main.write(str(val) + ',')

                # plot the distributions as well as the amplitudes at the grid points
                if plotting:
                    plt.figure()
                    plt.plot(x, p, '-k', label='QCT')
                    plt.plot(grid, p_loc_avg_on_grid, '.g', label='Amplitudes')
                    plt.xlabel(r"$E_{\mathrm{trans}}$" + ' [eV]')
                    plt.ylabel('Probability')
                    plt.figtext(0.70, 0.65, r"$T_{\mathrm{trans}}$ = " +
                                str(temperatures[0]) + ' K')
                    plt.title(r"$P(E_{\mathrm{trans}})$")
                    plt.legend()
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()

            # similarly for the reactant vibrational state distributions
            elif filename == 'rv':
                grid = grid_v_reac

                # n_max = 1
                p_loc_avg = []
                for j in range(len(p)):
                    if j-1 < 0 or j+1 > (len(p)-1):
                        p_loc_avg.append(p[j])
                    else:
                        p_loc_avg.append(np.mean(p[j-1:j+1+1]))

                p_loc_avg_on_grid = []
                for ele in grid:
                    val = p_loc_avg[list(x).index(ele)]
                    p_loc_avg_on_grid.append(val)
                    txt_file_main.write(str(val) + ',')

                if plotting:
                    plt.figure()
                    plt.plot(x, p, '-k', label='QCT')
                    plt.plot(grid, p_loc_avg_on_grid, '.g', label='Amplitudes')
                    plt.xlabel(r"$\it{v}$")
                    plt.ylabel('Probability')
                    plt.figtext(0.70, 0.65, r"$T_{\mathrm{rovib}}$ = " +
                                str(temperatures[1]) + ' K')
                    plt.title(r"$P(\it{v})$")
                    plt.legend()
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()

            # reactant rotational state distributions
            elif filename == 'rj':
                grid = grid_j_in

                # n_max = 9
                p_loc_avg = []
                for j in range(len(p)):
                    if j-1 < 0 or j+1 > (len(p)-1):
                        p_loc_avg.append(p[j])
                    elif j-2 < 0 or j+2 > (len(p)-2):
                        p_loc_avg.append(np.mean(p[j-1:j+1+1]))
                    elif j-3 < 0 or j+3 > (len(p)-3):
                        p_loc_avg.append(np.mean(p[j-2:j+2+1]))
                    elif j-4 < 0 or j+4 > (len(p)-4):
                        p_loc_avg.append(np.mean(p[j-3:j+3+1]))
                    elif j-5 < 0 or j+5 > (len(p)-5):
                        p_loc_avg.append(np.mean(p[j-4:j+4+1]))
                    elif j-6 < 0 or j+6 > (len(p)-6):
                        p_loc_avg.append(np.mean(p[j-5:j+5+1]))
                    elif j-7 < 0 or j+7 > (len(p)-7):
                        p_loc_avg.append(np.mean(p[j-6:j+6+1]))
                    elif j-8 < 0 or j+8 > (len(p)-8):
                        p_loc_avg.append(np.mean(p[j-7:j+7+1]))
                    elif j-9 < 0 or j+9 > (len(p)-9):
                        p_loc_avg.append(np.mean(p[j-8:j+8+1]))
                    else:
                        p_loc_avg.append(np.mean(p[j-9:j+9+1]))

                p_loc_avg_on_grid = []
                for ele in grid:
                    val = p_loc_avg[list(x).index(ele)]
                    p_loc_avg_on_grid.append(val)
                    txt_file_main.write(str(val) + ',')

                if plotting:
                    plt.figure()
                    plt.plot(x, p, '-k', label='QCT')
                    plt.plot(grid, p_loc_avg_on_grid, '.g', label='Amplitudes')
                    plt.xlabel(r"$\it{j}$")
                    plt.ylabel('Probability')
                    plt.figtext(0.70, 0.65, r"$T_{\mathrm{rovib}}$ = " +
                                str(temperatures[1]) + ' K')
                    plt.title(r"$P(\it{j})$")
                    plt.legend()
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()

            # product relative translational energy distributions
            elif filename == 'pe':
                grid = grid_Etrans_prod

                # n_max = 3
                avg_p = []
                for j in range(len(p)):
                    if j-1 < 0 or j+1 > (len(p)-1):
                        avg_p.append(p[j])
                    elif j-2 < 0 or j+2 > (len(p)-2):
                        avg_p.append(np.mean(p[j-1:j+1+1]))
                    elif j-3 < 0 or j+3 > (len(p)-3):
                        avg_p.append(np.mean(p[j-2:j+2+1]))
                    else:
                        avg_p.append(np.mean(p[j-3:j+3+1]))

                p_loc_avg_on_grid = []
                for ele in grid:
                    val = avg_p[list(x).index(ele)]
                    p_loc_avg_on_grid.append(val)
                    txt_file_main.write(str(val) + ',')

                if plotting:
                    plt.figure()
                    plt.plot(x, p, '-k', label='QCT')
                    plt.plot(grid, p_loc_avg_on_grid, '.g', label='Amplitudes')
                    plt.xlabel(r"$E_{\mathrm{trans}}'$" + ' [eV]')
                    plt.ylabel('Probability')
                    plt.legend()
                    plt.tight_layout()
                    plt.title(r"$P(E_{\mathrm{trans}}')$")
                    pdf.savefig()
                    plt.close()

            # product vibrational state distributions
            elif filename == 'pv':
                grid = grid_v_prod

                # n_max = 2
                avg_p = []
                for j in range(len(p)):
                    if j-1 < 0 or j+1 > (len(p)-1):
                        avg_p.append(p[j])
                    elif j-2 < 0 or j+2 > (len(p)-2):
                        avg_p.append(np.mean(p[j-1:j+1+1]))
                    else:
                        avg_p.append(np.mean(p[j-2:j+2+1]))

                p_loc_avg_on_grid = []
                for ele in grid:
                    val = avg_p[list(x).index(ele)]
                    p_loc_avg_on_grid.append(val)
                    txt_file_main.write(str(val) + ',')

                if plotting:
                    plt.figure()
                    plt.plot(x, p, '-k', label='QCT')
                    plt.plot(grid, p_loc_avg_on_grid, '.g', label='Amplitudes')
                    plt.xlabel(r"$\it{v'}$")
                    plt.ylabel('Probability')
                    plt.title(r"$P(\it{v'})$")
                    plt.legend()
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()

            # product rotational state distributions
            elif filename == 'pj':
                grid = grid_j_prod

                # n_max = 10
                avg_p = []
                for j in range(len(p)):
                    if j-1 < 0 or j+1 > (len(p)-1):
                        avg_p.append(p[j])
                    elif j-2 < 0 or j+2 > (len(p)-2):
                        avg_p.append(np.mean(p[j-1:j+1+1]))
                    elif j-3 < 0 or j+3 > (len(p)-3):
                        avg_p.append(np.mean(p[j-2:j+2+1]))
                    elif j-4 < 0 or j+4 > (len(p)-4):
                        avg_p.append(np.mean(p[j-3:j+3+1]))
                    elif j-5 < 0 or j+5 > (len(p)-5):
                        avg_p.append(np.mean(p[j-4:j+4+1]))
                    elif j-6 < 0 or j+6 > (len(p)-6):
                        avg_p.append(np.mean(p[j-5:j+5+1]))
                    elif j-7 < 0 or j+7 > (len(p)-7):
                        avg_p.append(np.mean(p[j-6:j+6+1]))
                    elif j-8 < 0 or j+8 > (len(p)-8):
                        avg_p.append(np.mean(p[j-7:j+7+1]))
                    elif j-9 < 0 or j+9 > (len(p)-9):
                        avg_p.append(np.mean(p[j-8:j+8+1]))
                    elif j-10 < 0 or j+10 > (len(p)-10):
                        avg_p.append(np.mean(p[j-9:j+9+1]))
                    else:
                        avg_p.append(np.mean(p[j-10:j+10+1]))

                p_loc_avg_on_grid = []
                for ele in grid:
                    val = avg_p[list(x).index(ele)]
                    p_loc_avg_on_grid.append(val)
                    txt_file_main.write(str(val) + ',')

                if plotting:
                    plt.figure()
                    plt.plot(x, p, '-k', label='QCT')
                    plt.plot(grid, p_loc_avg_on_grid, '.g', label='Amplitudes')
                    plt.xlabel(r"$\it{j'}$")
                    plt.ylabel('Probability')
                    plt.title(r"$P(\it{j'})$")
                    plt.legend()
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()

        txt_file_main.write('\n')

if plotting:
    pdf.close()

# write the input for the data sets from the training and validation sets in a separate file
with open("input_and_reference_all.txt", "r") as txt_file:
    lines = txt_file.readlines()
with open("input_and_reference_train_valid.txt", "w") as txt_file:
    for i in np.subtract(train_valid_indices, start_index):
        txt_file.write(lines[i])
