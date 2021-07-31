Since the previous model showed a bias towards linear trajectories, I tried doubling the length of the trajectory to include sufficient training examples with circular nature. However, the bias seems intact, as is demonstrated by the following histogram: 

![Histogram](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance_same_start_double_length/histogram.jpg)

To demonstrate this further, the following two trajectories were used: 

![Trajectory1](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance_same_start_double_length/trajectory1/trajectory.jpg)
![Trajectory2](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance_same_start_double_length/trajectory2/trajectory.jpg)

The predictions on these two trajectories were as follows:

![Trajectory1_Pred](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance_same_start_double_length/trajectory1/trajectory_predictions.jpg)
![Trajectory2_Pred](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance_same_start_double_length/trajectory2/trajectory_predictions.jpg)

The R<sup>2</sup> values for the two trajectories were -1.814 and -0.293 respectively. 
