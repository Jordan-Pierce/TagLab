# TagLab: an image segmentation tool oriented to marine data analysis

| &nbsp; [Software Requirements](#software-requirements) &nbsp; | &nbsp; [Install](#installing-taglab) &nbsp; | &nbsp; [Update](#updating-taglab) &nbsp; | &nbsp; [Citation](#citation) &nbsp; |


## Introduction

TagLab was created to support the activity of annotation and extraction of statistical data from ortho-maps of benthic communities. The tool includes different types of CNN-based segmentation networks specially trained for agnostic (relative only to contours) or semantic (also related to species) recognition of corals. TagLab is an ongoing project of the  [Visual Computing Lab](https://vcg.isti.cnr.it).

![ScreenShot](screenshot.jpg)

#### Warning ⚠️

This is a developmental fork from the original repository; changes are made here, like, really frequently... proceed with caution.

#### ✨ New Features:
- 10/2023
  - Install script for Windows using [Anaconda](https://docs.conda.io/projects/miniconda/en/latest/); see Installation instructions below
  - [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) 
    - [`Predictor` for one-click segmentations](https://www.youtube.com/watch?v=3dMo2NO1iXQ)
    - [`Generator` for automatic segmentations](https://www.youtube.com/watch?v=3dMo2NO1iXQ)
- 12/2023
  - Width and Height of current view is shown in top-right (updates w/ zoom)
  - Users can mark grid cells as complete from within SAM and pos/neg (RITM) tools
  - Working area for SAM and pos/neg (RITM) tools will show you the dimensions of the working area
- 02/2024
  - [`iView`](https://www.youtube.com/watch?v=H3AjshATPy0) using Metashape API to view corresponding images of orthomosaic; requires Professional license
  - Fixed `Refine` tool bug
  - Point annotations (**all credit goes to original TagLab authors**)
    - Point sampling select work area (shows pixel and metric space)
    - Basic Import / Export of point annotations in CoralNet Format
- 03/2024
  - Integrated `Upload` and `API` tools from [`CoralNet Toolbox`](https://github.com/Jordan-Pierce/CoralNet-Toolbox) into `TagLab` to allow for `CoralNet` models to make predictions on point annotations in orthomosaic.

We are working hard to create a website with detailed instructions about TagLab. Stay tuned(!)


## Software Requirements


TagLab runs on __Linux__, __Windows__, and __MacOS__. To run TagLab, the main requirement is just __64bit Python 3.8.x, 3.9.x or 3.10.x__.

GPU accelerated computations are not supported on MacOS and on any machine that has not an NVIDIA graphics card.
To use them, you'll need to install the __NVIDIA CUDA Toolkit__, versions 10.2, 11.3, 11.6 and 11.7 are supported.
If you don't have a NVida graphics card (or if you use MacOS), CPU will be used.

## Installing TagLab

To install TagLab on Windows using [Anaconda](https://docs.conda.io/projects/miniconda/en/latest/), follow the
instructions below. If CUDA is not installed on your computer, include `cpu` when running the install script:
```python
# Anaconda Command Prompt

# Create a conda environment for TagLab
conda create --name taglab python=3.8 -y

# Activate the environment
conda activate taglab

# Within the TagLab folder, run the install_conda_windows.py script
python install_conda_windows.py [cpu]

# Run TagLab
python Taglab.py
```
For further instructions on installing TagLab see below.

### Step 1: Install Anaconda

1. **Download Anaconda:**
   - Go to the [Anaconda website](https://www.anaconda.com/products/distribution).
   - Download the appropriate version for your Windows system (e.g., 64-bit or 32-bit).
   - Follow the installation instructions provided on the website.

### Step 2: Open Anaconda Command Prompt

1. **Search for Anaconda Command Prompt:**
   - Press the `Windows` key on your keyboard.
   - Type "Anaconda Command Prompt" and press `Enter`.

### Step 3: Create a Conda Environment for TagLab

1. **Create a new conda environment:**
   - In the Anaconda Command Prompt, type the following command and press `Enter`:
     ```bash
     conda create --name taglab python=3.8 -y
     ```
   - This command creates a new conda environment named "taglab" with Python version 3.8.

### Step 4: Activate the Conda Environment

1. **Activate the conda environment:**
   - Type the following command and press `Enter`:
     ```bash
     conda activate taglab
     ```
   - Your command prompt should now show that you are in the "taglab" environment.

### Step 5: Download TagLab and Install Dependencies

1. **Navigate to the TagLab folder:**
   - Download this TagLab fork .
   - Extract the contents of the downloaded zip file to a location on your computer.
   - Open the Anaconda Command Prompt and navigate to the folder where TagLab is extracted, using the `cd` command. For example:
     ```bash
     cd path\to\Taglab
     ```

2. **Run the installation script:**
   - Type the following command and press `Enter`:
     ```bash
     python install_conda_windows.py [cpu]
     ```
     - Replace `[cpu]` with nothing if CUDA is installed on your computer, or include `cpu` if CUDA is not installed.
     - This script installs the necessary dependencies for TagLab.

### Step 6: Run TagLab

1. **Run TagLab:**
   - After the installation is complete, you can run TagLab using the following command:
     ```bash
     python Taglab.py
     ```
   - This command starts TagLab, and you should now be able to use the tool within the activated conda environment.

# Citation

If you use TagLab, please cite it.

```
@article{TagLab,
	author = {Pavoni, Gaia and Corsini, Massimiliano and Ponchio, Federico and Muntoni, Alessandro and Edwards, Clinton and Pedersen, Nicole and Sandin, Stuart and Cignoni, Paolo},
	title = {TagLab: AI-assisted annotation for the fast and accurate semantic segmentation of coral reef orthoimages},
	year = {2022},
	journal = {Journal of Field Robotics},
	volume = {39},
	number = {3},
	pages = {246 – 262},
	doi = {10.1002/rob.22049}
}
```