This section of the project aims to solve the same problem as the original DeepMind paper. However, instead of training the network directly on place cell and head direction cell distributions, I train it just on the vector to the starting location (home location). More precisely, I ask the agent to predict the distance from its current position to its home location, and the angle of the direct vector between these two locations. I am not constraining the network to use the place cell representation but allowing it to use any representation.

This section involves a lot of experimentation, hence the huge number of sub-directories. I have added README files for the key sub-directories. 