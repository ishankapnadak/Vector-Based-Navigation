An implementation where the trajectory fed is used to monitor the LSTM activity while the agent is moving and when it stops midway. For this part, I have used a trajectory in which the agent moves for 400 timesteps and remains still for the next 400 timesteps. The LSTM activity is sampled every 100 timesteps and is plotted as a GIF (to visualise its evolution). It appears that the LSTM activity remains constant once the agent stops moving.

![LSTM Activity](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/Old%20Supervised/Specialised/lstm_activity/activity.gif)
