from   scipy.io   import  loadmat
import numpy      as      np
import pandas     as      pd
from pandasgui import show
from matplotlib import pyplot as plt


def view_groundtruth():
    matrixVar = loadmat( "experiments/spikesorting_interp/data/truth.mat" )

    # matrixVar = loadmat( "experiments/spikesorting_interp/data/sparsity_level.mat" )
    datasets = {}
    for dataset in matrixVar:
        if dataset == '__header__' or dataset == '__version__' or dataset == '__globals__':
            print("Skipping", dataset, matrixVar[dataset])
            continue
        print(dataset, matrixVar[dataset].shape)
        datasets[dataset] = matrixVar[dataset].flatten() #pd.DataFrame(matrixVar[dataset].flatten())
        print(datasets[dataset].shape)

    # print(datasets['idx'])
    print(datasets['true_ts'][0])
    print(datasets['true_ts'][1])
    print(datasets['true_ts'][-2])
    print(datasets['true_ts'][-1])

    for i in range(len(datasets['true_ts'])):
        plt.axvline(x=datasets['true_ts'][i], color='r', linestyle='--')

    return datasets
    # print(datasets['signal'][0].shape)

    # max_len = 0
    # for i in range(0,5087,10):
    #     if len(datasets['signal'][i]) > max_len:
    #         max_len = len(datasets['signal'][i])
    #         print(i, max_len)
    #     centering = datasets['idx'][i][len(datasets['idx'][i])//2][0]
    #     # centering = datasets['idx'][i][0][0] + len(datasets['idx'][i])//2
    #     plt.plot((datasets['idx'][i].astype(int) - int(centering))/1e4, datasets['signal'][i])
    #     # print(datasets['idx'][i] - centering)


    # # print(datasets['signal'])

    # # plt.plot(datasets['idx'], datasets['signal'])
    # plt.show()

def view_signal():
    matrixVar = loadmat( "experiments/spikesorting_interp/data/data.mat" )

    # matrixVar = loadmat( "experiments/spikesorting_interp/data/sparsity_level.mat" )
    datasets = {}
    for dataset in matrixVar:
        if dataset == '__header__' or dataset == '__version__' or dataset == '__globals__':
            print("Skipping", dataset, matrixVar[dataset])
            continue
        print(dataset, matrixVar[dataset].shape)
        datasets[dataset] = matrixVar[dataset].flatten() #pd.DataFrame(matrixVar[dataset].flatten())
        print(datasets[dataset].shape)

    # print(datasets['idx'])
    print(datasets['idx'][0].shape)
    print(datasets['signal'][0].shape)

    max_len = 0
    for i in range(0,5087):
        if len(datasets['signal'][i]) > max_len:
            max_len = len(datasets['signal'][i])
            print(i, max_len)
        centering = datasets['idx'][i][len(datasets['idx'][i])//2][0]
        # centering = datasets['idx'][i][0][0] + len(datasets['idx'][i])//2
        # plt.plot((datasets['idx'][i].astype(int) - int(centering))/1e4, datasets['signal'][i])
        plt.plot(datasets['idx'][i].astype(int), datasets['signal'][i])
        # print(datasets['idx'][i] - centering)


    # print(datasets['signal'])

    # plt.plot(datasets['idx'], datasets['signal'])

    return datasets


# view_groundtruth()
# view_signal()
# plt.show()

def view_sim_hdf5():
    filename = "experiments/sim/data/T_3_noise_0.1_num_50_0.hdf5"
    import h5py
    with h5py.File(filename,'r') as f:
        data = f['data']
        print(data.shape)
        fs = data.attrs['fs']
        indices = data.attrs['indices']
        plt.plot(np.arange(data.shape[0])/fs, np.array(data))
        colors = ['r', 'g', 'b', 'y']
        for i in range(len(indices)):
            for j in range(len(indices[i])):
                plt.axvline(x=indices[i, j], color=colors[i], linestyle='--')

    plt.show()

view_sim_hdf5()

# print(len(matrixVar))
# datasets = {}
# for dataset in matrixVar:
#     if dataset == '__header__' or dataset == '__version__' or dataset == '__globals__':
#         print("Skipping", dataset, matrixVar[dataset])
#         continue
#     print(dataset)
#     # datasets[dataset] = pd.DataFrame(matrixVar[dataset].flatten())
#     datasets[dataset] = pd.DataFrame(matrixVar[dataset].flatten())
#     print(datasets[dataset].shape)
# # exit()
# # exit()

# # Do whatever data manipulation you need here
# # Let's do a simple transpose for the sake of the example.
# # mainpulatedData = np.transpose(matrixVar)

# # Do more stuff here if needed
# # ...
# # print(matrixVar)
# print( 'Done processing' )
# show(datasets['idx'])
