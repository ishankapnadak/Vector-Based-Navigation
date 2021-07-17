Trained only on distance, with three linear layers and intermediate tanh activations. The following mean loss was obtained:

![Mean Loss](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance/mean_loss.jpg)

A histogram of predicted distances was plotted for 400 trajectories (each 400 timesteps long):

![Histogram](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance/histogram.jpg)

The network predicts a near constant value and sort of resets after every batch size (100 timesteps). 
