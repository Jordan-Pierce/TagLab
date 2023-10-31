import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path
import importlib.util as importutil
import urllib.request

import re

osused = platform.system()
if osused != 'Linux' and osused != 'Windows' and osused != 'Darwin':
    raise Exception("Operative System not supported")

# Python version
python_v = f"{sys.version_info[0]}{sys.version_info[1]}"
python_sub_v = int(sys.version_info[1])

# check python version
if python_sub_v < 8 or python_sub_v > 10:
    raise Exception(f"Python 3.{python_sub_v} not supported. "
                    f"Please see https://github.com/cnr-isti-vclab/TagLab/wiki/Install-TagLab")

# manage torch
something_wrong_with_nvcc = False
flag_install_pythorch_cpu = False
nvcc_version = ''
torch_package = 'torch'
torchvision_package = 'torchvision'
torch_extra_argument1 = ''
torch_extra_argument2 = ''

# Windows CUDA path, since nvcc --version is unreliable
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"

# if the user wants to install cpu torch
if len(sys.argv) == 2 and sys.argv[1] == 'cpu':
    flag_install_pythorch_cpu = True

# Get nvcc version
if osused == 'Darwin':
    # For Mac, use CPU
    flag_install_pythorch_cpu = True
    print('NVCC not supported on MacOS. Installing cpu version automatically...')

# If user wants GPU pytorch
elif not flag_install_pythorch_cpu:

    # Check that cuda is in program files
    if not os.path.exists(cuda_path):
        print(f"Could not find {cuda_path}")
        print("If CUDA is not installed, use: 'python install_w_conda.py cpu' instead")
        sys.exit(1)

    # For Windows, find CUDA versions in Program Files
    cuda_versions = os.listdir(cuda_path)

    # List containing CUDA versions installed (properly)
    cuda_versions_installed = []

    for cuda_version in cuda_versions:
        cuda_folder = f"{cuda_path}\\{cuda_version}\\bin"
        # Presence of bin indicates that it was probably installed properly
        if os.path.exists(cuda_folder):
            cuda_versions_installed.append(cuda_version)

    if cuda_versions:
        # Find the newest version that was installed properly
        nvcc_version = re.sub(r'[a-zA-Z]', '', max(cuda_versions_installed))

        # Install NVCC version using conda
        conda_exe = shutil.which('conda')

        if conda_exe:
            # Create a conda command
            conda_command = [conda_exe, "install", "-c", f"nvidia/label/cuda-{nvcc_version}.0", "cuda-nvcc", "-y"]

            # Run the conda command
            subprocess.run(conda_command, check=True)
        else:
            print("Conda executable not found. Make sure Conda is installed and in your system's PATH.")
            sys.exit(1)
    else:
        something_wrong_with_nvcc = True
        nvcc_version = ""

    # Install pytorch using nvcc version
    if '9.2' in nvcc_version:
        nvcc_version = '9.2'
        print('Torch 1.7.1 for CUDA 9.2')
        torch_package += '==1.7.1+cu92'
        torchvision_package += '==0.8.2+cu92'
        torch_extra_argument1 = '-f'
        torch_extra_argument2 = 'https://download.pytorch.org/whl/torch_stable.html'
    elif nvcc_version == '10.1':
        print('Torch 1.7.1 for CUDA 10.1')
        torch_package += '==1.7.1+cu101'
        torchvision_package += '==0.8.2+cu101'
        torch_extra_argument1 = '-f'
        torch_extra_argument2 = 'https://download.pytorch.org/whl/torch_stable.html'
    elif nvcc_version == '10.2':
        print('Torch 1.11.0 for CUDA 10.2')
        torch_package += '==1.11.0+cu102'
        torchvision_package += '==0.12.0+cu102'
        torch_extra_argument1 = '--extra-index-url'
        torch_extra_argument2 = 'https://download.pytorch.org/whl/cu102'
    elif '11.0' in nvcc_version:
        print('Torch 1.7.1 for CUDA 11.0')
        torch_package += '==1.7.1+cu110'
        torchvision_package += '0.8.2+cu110'
        torch_extra_argument1 = '-f'
        torch_extra_argument2 = 'https://download.pytorch.org/whl/torch_stable.html'
    elif '11.1' in nvcc_version:
        print('Torch 1.8.0 for CUDA 11.1')
        torch_package += '==1.8.0+cu111'
        torchvision_package += '==0.9.0+cu111'
        torch_extra_argument1 = '-f'
        torch_extra_argument2 = 'https://download.pytorch.org/whl/torch_stable.html'
    elif '11.3' in nvcc_version:
        print('Torch 1.12.1 for CUDA 11.3')
        torch_package += '==1.12.1+cu113'
        torchvision_package += '==0.13.1+cu113'
        torch_extra_argument1 = '--extra-index-url'
        torch_extra_argument2 = 'https://download.pytorch.org/whl/cu113'
    elif '11.6' in nvcc_version:
        print('Torch 1.13.1 for CUDA 11.6')
        torch_package += '==1.13.1+cu116'
        torchvision_package += '==0.14.1+cu116'
        torch_extra_argument1 = '--extra-index-url'
        torch_extra_argument2 = 'https://download.pytorch.org/whl/cu116'
    elif '11.7' in nvcc_version:
        print('Torch 1.13.1 for CUDA 11.7')
        torch_package += '==1.13.1+cu117'
        torchvision_package += '==0.14.1+cu117'
        torch_extra_argument1 = '--extra-index-url'
        torch_extra_argument2 = 'https://download.pytorch.org/whl/cu117'
    elif '11.8' in nvcc_version:
        print("Torch 2.0.0 for CUDA 11.8")
        torch_package += '==2.0.0+cu118'
        torchvision_package += '==0.15.1+cu118'
        torch_extra_argument1 = '--extra-index-url'
        torch_extra_argument2 = 'https://download.pytorch.org/whl/cu118'
    elif '12.1' in nvcc_version or (nvcc_version and not something_wrong_with_nvcc):
        print("Torch 2.1.0 for CUDA")
        torch_extra_argument1 = '--index-url'
        torch_extra_argument2 = 'https://download.pytorch.org/whl/cu121'

    # if the user tried to run the installer but there were issues on finding a supported
    if something_wrong_with_nvcc == True and flag_install_pythorch_cpu == False:
        ans = input('Something is wrong with NVCC. '
                    'Do you want to install the CPU version of pytorch? [Y/n]')
        if ans.lower().strip() == "y":
            flag_install_pythorch_cpu = True
        else:
            raise Exception('Installation aborted. '
                            'Install a proper NVCC version or set the pytorch CPU version.')

# somewhere before, this flag has been set to True and the
# user choose to install the cpu torch version
if flag_install_pythorch_cpu == True:
    print('Torch will be installed in its CPU version.')
    if osused != 'Darwin':
        torch_extra_argument1 = '--extra-index-url'
        torch_extra_argument2 = 'https://download.pytorch.org/whl/cpu'

# manage gdal
gdal_version = ''

if osused == 'Linux':
    result = subprocess.getstatusoutput('gdal-config --version')
    output = result[1]
    rc = result[0]
    if rc != 0:
        print('Trying to install libgdal-dev...')
        from subprocess import STDOUT, check_call
        import os

        try:
            check_call(['sudo', 'apt-get', 'install', '-y', 'libgdal-dev'],
                       stdout=open(os.devnull, 'wb'), stderr=STDOUT)
        except:
            raise Exception('Impossible to install libgdal-dev. '
                            'Please install manually libgdal-dev before running this script.'
                            '\nInstallation aborted.')

        result = subprocess.getstatusoutput('gdal-config --version')
        output = result[1]
        rc = result[0]
    if rc == 0:
        gdal_version = output
        print('GDAL version installed: ' + output)
    else:
        raise Exception('Impossible to access to gdal-config binary.\nInstallation aborted.')

    print('Trying to install libxcb-xinerama0...')
    from subprocess import STDOUT, check_call
    import os

    try:
        check_call(['sudo', 'apt-get', 'install', '-y', 'libxcb-xinerama0'],
                   stdout=open(os.devnull, 'wb'), stderr=STDOUT)
    except:
        print('Impossible to install libxcb-xinerama0. '
              'If TagLab does not start, please install manually libxcb-xinerama0.')

elif osused == 'Darwin':
    result = subprocess.getstatusoutput('gdal-config --version')
    output = result[1]
    rc = result[0]
    if rc != 0:
        print('Trying to install gdal...')
        from subprocess import STDOUT, check_call
        import os

        try:
            check_call(['brew', 'install', 'gdal'],
                       stdout=open(os.devnull, 'wb'), stderr=STDOUT)
        except:
            raise Exception('Impossible to install gdal through homebrew. '
                            'Please install manually gdal before running this script.'
                            '\nInstallation aborted.')

        result = subprocess.getstatusoutput('gdal-config --version')
        output = result[1]
        rc = result[0]
    if rc == 0:
        gdal_version = output
        print('GDAL version installed: ' + output)
    else:
        raise Exception('Impossible to access to gdal-config binary.\nInstallation aborted.')

gdal_package = 'gdal==' + gdal_version

# build coraline
if osused != 'Windows':
    try:
        out = subprocess.check_output(['cmake', '--version'])
        if out[0] != 0:
            if osused == 'Darwin':
                print('Trying to install cmake...')
                from subprocess import STDOUT, check_call
                import os

                try:
                    check_call(['brew', 'install', 'cmake'],
                               stdout=open(os.devnull, 'wb'), stderr=STDOUT)
                except:
                    raise Exception('Impossible to install cmake through homebrew. '
                                    'Please install manually cmake before running this script.'
                                    '\nInstallation aborted.')
            elif osused == 'Linux':
                print('Trying to install cmake...')
                from subprocess import STDOUT, check_call
                import os

                try:
                    check_call(['sudo', 'apt-get', 'install', '-y', 'cmake'],
                               stdout=open(os.devnull, 'wb'), stderr=STDOUT)
                except:
                    raise Exception('Impossible to install cmake. '
                                    'Please install manually cmake before running this script.'
                                    '\nInstallation aborted.')
        os.chdir('coraline')
        result = subprocess.getstatusoutput('cmake .')
        if result[0] == 0:
            result = subprocess.getstatusoutput('make')
            if result[0] == 0:
                print('Coraline built correctly.')
                os.chdir('..')
            else:
                raise Exception('Error while building Coraline library.\nInstallation aborted.')
        else:
            raise Exception('Error while configuring Coraline library.\nInstallation aborted.')
    except OSError:
        raise Exception(
            'Cmake not found. Coraline library cannot be compiled. Please install cmake '
            'first.\nInstallation aborted.')

# requirements needed by TagLab
install_requires = [
    'wheel',
    'pyqt5',
    'scikit-image',
    'scikit-learn',
    'pandas',
    'opencv-python',
    'matplotlib',
    'albumentations',
    'shapely',
]

# if on windows, first install the msvc runtime, pycocotools
if osused == 'Windows':
    install_requires.insert(0, 'msvc-runtime')

if osused == 'Windows' and python_sub_v < 9:
    install_requires.append('pycocotools-windows')
else:
    # jesus, take the wheel
    install_requires.append('pycocotools')

# installing all the packages
for package in install_requires:

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    except Exception as e:
        print(f"There was an issue installing the necessary packages.\n{e}")

        if "pycocotools" in e:
            print(f"The error came from installing {install_requires[-1]}")
            print(f"If you're not already, please try using a conda environment with python 3.8")

        sys.exit(1)


# torch and torchvision
list_args = [sys.executable, "-m", "pip", "install", torch_package, torchvision_package]
if torch_extra_argument1 != "":
    list_args.extend([torch_extra_argument1, torch_extra_argument2])

# installing torch, torchvision
subprocess.check_call(list_args)

# gdal and rasterio
if osused != 'Windows':
    subprocess.check_call([sys.executable, "-m", "pip", "install", gdal_package])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'rasterio'])
else:
    # Locally stored wheels
    base_url = './packages/'
    # compute rasterio and gdal urls download
    gdal_win_version = '3.4.3'
    filename_gdal = 'gdal-' + gdal_win_version + '-cp' + python_v + '-cp' + python_v
    filename_gdal += '-win_amd64.whl'
    base_url_gdal = base_url + filename_gdal

    if not os.path.exists(base_url_gdal):
        raise Exception(f"Could not find {base_url_gdal}; aborting")

    rasterio_win_version = '1.2.10'
    filename_rasterio = 'rasterio-' + rasterio_win_version + '-cp' + python_v + '-cp' + python_v
    filename_rasterio += '-win_amd64.whl'
    base_url_rasterio = base_url + filename_rasterio

    if not os.path.exists(base_url_rasterio):
        raise Exception(f"Could not find {base_url_rasterio}; aborting")

    # see if rasterio and gdal are already installed
    try:
        gdal_is_installed = importutil.find_spec("osgeo.gdal")
    except:
        gdal_is_installed = None

    if gdal_is_installed is not None:
        import osgeo.gdal

        print("GDAL ", osgeo.gdal.__version__, " is installed. "
              "Version ", gdal_win_version, "is required.")
    else:
        # retrieve GDAL from TagLab website
        print('GET GDAL FROM URL: ' + base_url_gdal)

        # install gdal from packages
        subprocess.check_call([sys.executable, "-m", "pip", "install", base_url_gdal])

    try:
        rasterio_is_installed = importutil.find_spec("rasterio")
    except:
        rasterio_is_installed = None

    # if so, check versions
    if rasterio_is_installed is not None:
        import rasterio

        print("RASTERIO ", rasterio.__version__, " is installed. "
              "Version ", rasterio_win_version, " is required.")
    else:
        # retrieve rasterio from TagLab website
        print('GET RASTERIO FROM URL: ' + base_url_rasterio)

        # install rasterio
        subprocess.check_call([sys.executable, "-m", "pip", "install", base_url_rasterio])

# Install Numpy (installing before raises warnings about versions conflicts)
command = [sys.executable, "-m", "pip", "install", "numpy"]
subprocess.run(command, check=True)

# Install SAM
command = [sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/segment-anything.git"]
subprocess.run(command, check=True)

# check for other networks
print('Downloading networks...')
from os import path
import urllib.request

this_directory = path.abspath(path.dirname(__file__))

# TagLab Weights
base_url = 'http://taglab.isti.cnr.it/models/'
net_file_names = ['dextr_corals.pth',
                  'deeplab-resnet.pth.tar',
                  'ritm_corals.pth',
                  'pocillopora.net',
                  'porites.net',
                  'pocillopora_porite_montipora.net']

for net_name in net_file_names:
    filename_dextr_corals = 'dextr_corals.pth'
    net_file = Path('models/' + net_name)
    if not net_file.is_file():  # if file not exists
        try:
            url_dextr = base_url + net_name
            print('Downloading ' + url_dextr + '...')
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url_dextr, 'models/' + net_name)
        except:
            raise Exception("Cannot download " + net_name + ".")
    else:
        print(net_name + ' already exists.')


# SAM Weights
base_url = "https://dl.fbaipublicfiles.com/segment_anything/"
net_file_names = ["sam_vit_b_01ec64.pth",
                  "sam_vit_l_0b3195.pth",
                  "sam_vit_h_4b8939.pth"]

for net_name in net_file_names:
    path_dextr = f"models/{net_name}"
    if not os.path.exists(path_dextr):
        try:
            url_dextr = base_url + net_name
            print('Downloading ' + url_dextr + '...')
            # Send an HTTP GET request to the URL
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url_dextr, path_dextr)
            print(f"NOTE: Downloaded file successfully")
            print(f"NOTE: Saved file to {path_dextr}")
        except:
            raise Exception("Cannot download " + net_name + ".")

    else:
        print(net_name + ' already exists.')
