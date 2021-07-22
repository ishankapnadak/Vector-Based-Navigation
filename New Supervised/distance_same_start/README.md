Fixed the starting position of the trajectory. This resulted in the following activity map: 

![Activity](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance_same_start/activityMaps/neurons1.jpg)

The loss curve obtained was as follows:

![Mean Loss](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance_same_start/mean_loss.jpg)

Histogram:

![Histogram](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance_same_start/histogram.jpg)

The network predicts some sort of increasing distance. As such, it should work well on trajectories that move away progressively from the home location, but should not work well on trajectories that are circular in nature, or move closer to the home location at a later point in the trajectory. To verify this, I plotted two trajectories: (1) where the agent moves progressively away from the home location, and (2) where the agent comes closer to the home location later on. These two trajectories are plotted below:

![Trajectory 1](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance_same_start/trajectory1/trajectory1.jpg) ![Trajectory 2](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance_same_start/trajectory2/trajectory2.jpg)

The R<sup>2</sup> coefficient between the predicted distances and the actual distances for these two trajectories is 
1. 0.938
2. 0.023

This demonstrates that the performance of the network on the first trajectory is very good, but significantly deteriorates in the second trajectory. This is further demonstrated by the following two plots for the first and second trajectory respectively:

![Trajectory 1 Predictions](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance_same_start/trajectory1/trajectory1_predictions.jpg)

![Trajectory 2 Predictions](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance_same_start/trajectory2/trajectory2_predictions.jpg)
