# Glomeruli segmentation in renal biopsy images using Machine Learning (TF/Keras)

## Introduction

[Classic U-Net model](https://github.com/bnsreenu/python_for_microscopists)

RESULTS IMAGES

## Environment Set-up

### Creating a Conda Environment

1. Install conda on your device: installation guide for
   [Windows](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) |
   [Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
   By default, conda creates the _base_ environment. You can deactivate it with `conda deactivate base`
2. Create a new environment specifying Python version.

```bash
username:~$ conda create --name your_env_name python=3.8
```

**NOTE:** Any Python3 version is valid. Nonetheless, 3.8 or higher is recommended.

3. Activate the new environment. When an environment is active, the name within brackets will appear before your
   username.

```bash
username:~$ conda activate your_env_name 
(your_env_name) username:~$ 
```

### Project requirements

4. Use `pip` to install the list of required modules. They are listed in the `requirements.txt` file included in the
   parent directory.

```bash
(your_env_name) username:~$ pip install -r /path/to/repo/requirements.txt
```

<h3 id="openslide"><a target=_blank href=https://openslide.org/api/python/>OpenSlide for Python</a></h3>

#### Installation Steps

- **Windows**

1. Install OpenSlide API for Python:

```bash
username:~$ pip install openslide-python
```

2. Install [OpenSlide Source Binaries](https://openslide.org/download/), downloading the last version
   (**Windows Binaries** section). Unzip it in your desired directory. The folder structure can be seen below.

```bash
username:path-to-openslide-folder~$ tree 
├───bin/
│   ├───libopenslide-0.dll
│   └───...
├───include/
│   └───openslide/
├───lib/
└───licenses/
```

3. Create a new environment variable named `OPENSLIDE_PATH` that contains the path to the `bin/` directory. **This step
   is crucial**. You can follow this
   [guide](https://docs.oracle.com/en/database/oracle/machine-learning/oml4r/1.5.1/oread/creating-and-modifying-environment-variables-on-windows.html),
   or open a command prompt and type the following command (the `echo` command is just used to ensure that the variable has
   been succesfully saved):

```bash
C:/Users/user> setx OPENSLIDE_PATH "path/to/openslide/bin"
C:/Users/user> echo %OPENSLIDE_PATH%
```

4. Lastly, you have to include the following code snippet in those files where you want to use OpenSlide:

```python
import os
from sys import platform

_dll_path = os.getenv("OPENSLIDE_PATH")  
if _dll_path is not None:
    # Python >= 3.8
    with os.add_dll_directory(_dll_path):
        import openslide
```

- **Linux (Ubuntu)**

Install OpenSlide API for Python:

```bash
username:~$ pip install openslide-python
```

Install OpenSlide source:

```bash
username:~$ sudo apt-get install openslide-tools
```

Import OpenSlide to your Python script:

```python
import openslide
```

## Project Structure

### data

Datasets used for different tasks (training, test, etc.) should be contained inside this folder.

The following subfolder tree is expected:

```bash
data/
├───raw/
│   └─── # WSI's *.tif files  
└───segmenter/
    ├───HE/
    │   ├───ims/      # Images obtained from WSI's
    │   ├───gt/masks/ # Groundtruth masks 
    │   └───xml/      # [Optional] xml files with glomeruli coordinates
    ├───PAS/...
    └───PM/...
```

WSI in `*.tif` format are expected to be inside the `raw/` directory. 3200x3200px images extracted from them will be
placed in the corresponding staining folder contained in the `segmenter/` directory.

### models

This folder contains pre-trained weights for a Keras U-Net model. The name of this files is expected to follow a certain
naming format:

```
<model_name>-<staining>-<resize_ratio>-<date>.hdf5
```

### scripts

Every executable Python file can be found in this folder. Further details about the main scripts can be found in the
[How to run](#how-to-run) section.

### src

Source and utility files needed to run the project scripts.

## How to run

## Results

## Future work

## References

## Goals

- [ ]  Build a ML-based segmentation tool using [TF/Keras](https://www.tensorflow.org/overview) to find
  glomeruli in renal tissue [Whole-Slide Images](https://www.mbfbioscience.com/whole-slide-imaging) (WSI).
  At least,  a hit percentage of **95%** is required.
- [ ]  Develop a functional application for glomeruli classification from renal biopsy WSI. The final result
  might accepts as input a renal tissue WSI, and get as output the classification for the whole set of glomeruli
  contained in that image.
- [ ]  Compare different segmentation model architectures to extract conclusions about what works better for
  our case study: Classic U-Net [[1]](#1), [DoubleU-Net](https://arxiv.org/pdf/2006.04868.pdf), [U-Net++,](https://arxiv.org/pdf/1807.10165.pdf) belong others.

## Application stages

1. Renal tissue images use to have huge dimensions, being highly not recommended to directly process them. Cutting up
   these images into "patches" is an acceptable approach to reduce time and processing costs. We implement MATLAB
   OpenSlide library [[2]](#2) with this purpose, obtaining a set of _3200 x 3200 px_ images.
2. The obtained images can be employed to train a segmentation neural network architecture, as a U-Net, to find specific
   elements. In our case, the target are glomeruli (see Figure 1). As we are working with supervised learning, the model
   needs both the set of images and a ground-truth set of masks. This masks has been acquired manually.
3. Once the network has been trained and its weights is saved, the model will try to automatically find glomeruli in
   input images. Thinking about the last stage (classification), we have to build a dataset of glomeruli. Using the coordinates
   of each segmented glomerulus, a _600 x 600 px_ patch can be extracted for each of them.
4. The whole set of _600 x 600 px_ images will be fed to a multi-class classification network to detect whether a glomerulus
   has any disease or not.

![im](ims/stages.svg)
Figure 2: Application stages

**NOTE:**  Stages 1 and 3 work with [OpenSlide](https://openslide.org/) to work with Whole-Slide Images. This library
is C-native, but there exists APIs for MATLAB and Python, for instance. See the section
[OpenSlide for Python](#openslide-for-python).

![handcrafted](ims/handcrafted.png)
Figure 2: Obtaining handcrafted masks. The first step was to remark the limits of each glomerulus with an easily
perceptible color. Next, using the `manual_masks.py` script, binary masks are obtained.

## References

<a id="ref1">[1]</a>
Bhattiprolu, Sreenivas.
Python for microscopists ([GitHub repository](https://github.com/bnsreenu/python_for_microscopists))

<a id="2">[2]</a>
Daniel Forsberg (2022).
fordanic/openslide-matlab,
([GitHub](https://github.com/fordanic/openslide-matlab)). Retrieved February 1, 2022.
