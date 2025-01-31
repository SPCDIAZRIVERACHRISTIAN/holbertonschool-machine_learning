# Regularization

This directory is dedicated to learning about **regularization** in machine learning. Regularization is a technique used to prevent overfitting by adding constraints or modifications to the learning process.

## Learning Objectives
At the end of this project, you should be able to explain the following concepts to anyone without the help of Google:

### General
- What is **regularization**? What is its purpose?
- What are **L1** and **L2** regularization? What is the difference between the two methods?
- What is **dropout**?
- What is **early stopping**?
- What is **data augmentation**?
- How do you implement the above regularization methods in **Numpy** and **TensorFlow**?
- What are the **pros and cons** of the above regularization methods?

## Requirements

### General
- Allowed editors: `vi`, `vim`, `emacs`
- All files will be interpreted/compiled on **Ubuntu 20.04 LTS** using **Python 3.9**
- Files will be executed with **Numpy 1.25.2** and **TensorFlow 2.15**
- All files should end with a new line
- The first line of all Python scripts must be exactly:
  ```python
  #!/usr/bin/env python3
  ```
- A `README.md` file at the root of the project folder is mandatory
- Code should follow **pycodestyle** (version 2.11.1)
- All modules should have documentation:
  ```sh
  python3 -c 'print(__import__("my_module").__doc__)'
  ```
- All classes should have documentation:
  ```sh
  python3 -c 'print(__import__("my_module").MyClass.__doc__)'
  ```
- All functions (inside and outside a class) should have documentation:
  ```sh
  python3 -c 'print(__import__("my_module").my_function.__doc__)'
  python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'
  ```
- Unless otherwise noted, you are only allowed to import:
  ```python
  import numpy as np
  import tensorflow as tf
  ```
- You should not import any module unless it is being used
- All files must be executable
- The length of your files will be tested using `wc`
- When initializing layer weights, use:
  ```python
  tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))

