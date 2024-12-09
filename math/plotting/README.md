# Plotting Exercises

This directory contains a series of Python exercises that I will work through to understand the basics of data visualization and plotting with `matplotlib`. Each task builds on fundamental skills needed to create and customize various types of plots, from simple line charts to stacked bar graphs.

## Resources

Before I begin, I might find the following resources helpful to deepen my understanding:

- [Plot (graphics)](https://en.wikipedia.org/wiki/Plot_(graphics))
- [Scatter plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html)
- [Line chart](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)
- [Bar chart](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html)
- [Histogram](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html)
- [Pyplot tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)
- [matplotlib.pyplot documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html)
- [mplot3d tutorial](https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html)
- [Additional tutorials](https://matplotlib.org/stable/tutorials/index.html)

Some key `matplotlib.pyplot` functions and their purposes include:

- `plot()` - Create line plots
- `scatter()` - Create scatter plots
- `bar()` - Create bar charts
- `hist()` - Create histograms
- `xlabel()`, `ylabel()`, `title()` - Label axes and provide a title
- `subplot()`, `subplots()`, `subplot2grid()` - Create multiple plots in one figure
- `suptitle()` - Create a title for the entire figure
- `xscale()`, `yscale()` - Set scales (e.g., linear, log)
- `xlim()`, `ylim()` - Set axis limits

## Learning Objectives

By the end of these exercises, I expect to be able to:

- Explain what a plot is and why data visualization matters
- Differentiate between a scatter plot, line graph, bar graph, and histogram
- Understand what `matplotlib` is and why it’s commonly used in Python
- Plot data using `matplotlib` and customize the appearance of my plots
- Add labels, titles, and legends to plots
- Adjust axis scales and limits
- Plot multiple datasets on the same figure

## Requirements

- **Environment:**
  - My code will run on Ubuntu 20.04 LTS
  - I will use Python3 (version 3.9)
  - I will use `numpy` (version 1.25.2)
  - I will use `matplotlib` (version 3.8.3)

- **Code Standards:**
  - My code must pass `pycodestyle` (version 2.11.1) checks.
  - Each file will end with a new line.
  - Each file will start with `#!/usr/bin/env python3`
  - I will include proper documentation for modules, classes, and functions.
  - I will not add any extra imports unless noted.
  - All files must be executable.

- **Installation:**
  ```bash
  pip install --user matplotlib==3.8.3
  pip install --user Pillow==10.3.0
  sudo apt-get install python3-tk
