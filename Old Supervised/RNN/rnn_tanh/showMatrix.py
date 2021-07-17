import numpy as np
import os 
import matplotlib.pyplot as plt
import os
from scipy.signal import correlate2d
import tensorflow as tf

from ratSimulator import RatSimulator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def showMatrix(agent):
    #kernel = np.array(agent.rnn_cell._kernel.read_value())
    kernel, recurrent_to_grid, grid_to_place, grid_to_head = agent.sess.run([agent.rnn_cell._kernel, agent.W1, agent.W2, agent.W3])
    plt.imshow(kernel, cmap="inferno")
    plt.colorbar()
    plt.savefig('weight_matrix.jpg')
    plt.clf()
    #plt.show()

    plt.imshow(recurrent_to_grid, cmap="inferno")
    plt.colorbar()
    plt.savefig('recurrent_to_grid.jpg')
    plt.clf()
    print("-------------- Recurrent To Grid ------------------")
    print(recurrent_to_grid)
    print("---------------------------------------------------")

    plt.imshow(grid_to_place, cmap="inferno")
    plt.colorbar()
    plt.savefig('grid_to_place.jpg')
    plt.clf()
    print("-------------- Grid to Place ------------------")
    print(grid_to_place)
    print("-----------------------------------------------")

    plt.imshow(grid_to_head, cmap="inferno")
    plt.colorbar()
    plt.savefig('grid_to_head.jpg')
    plt.clf()
    print("-------------- Grid to Head ------------------")
    print(grid_to_head)
    print("----------------------------------------------")
