# Vector-Based Navigation in Artificial Agents

This repository contains my ongoing work under Prof. [Zoran Tiganj](https://homes.luddy.indiana.edu/ztiganj/index.html), Indiana University Bloomington. The intent of this project is to implement, and build on DeepMind's recent paper, [Vector-based navigation using grid-like representations in artificial agents](https://www.nature.com/articles/s41586-018-0102-6). I have used code from [Stefano Rosà's implementation](https://github.com/R-Stefano/Grid-Cells) of the same paper.

The first part of this project (contained in the folder 'Old Supervised') implements the original DeepMind paper, and analyses the performance as different parameters are varied. The second part of this project (contained in the folder 'New Supervised') takes a new approach to the same problem, training on vectors to the starting location (home location) rather than training on place cell and head direction cell distributions directly.
