# Model Odometry
## Description

This package is used for generating data on a humanoid robot and training a neural network to predict the step size of it.
The prediction is then used to calculate the odometry.

## Components

This package contains the following main components:

- Scripts: Mainly contains the scripts to process the data. There are also normalization and transformation components.
- Odometry Misc: The ros nodes for data collection are in there as well as the model odometry to be used to calculate the odometry on the robot.
- Model: The train scripts, network architectures and Optuna parameters are in here.

