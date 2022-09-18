# **SSD300 v1.1 For PyTorch on TPU**

This repository is a fork of https://github.com/NVIDIA/DeepLearningExamples modified to train the SSD network on Google Cloud TPU units.

### Setup

The following has been tested on a TPUv3-8 device. 

1. Clone the repository and install dependencies

   ```bash
   git clone https://github.com/jotron/DeepLearningExamplesTPU
   cd DeepLearningExamplesTPU/PyTorch/Detection/SSD
   
   pip install pycocotools
   pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger
   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda102
   ```

2. Download and preprocess the dataset with 

   ````bash
   export COCO_DIR=~/COCO
   ./download_dataset.sh $COCO_DIR
   ````

3. Train baseline model with

   You need to set the *DATA_DIR* variable in the script first.

   ```bash
   mkdir checkpoints
   bash ./examples/SSD300_1CORE_BS64.sh
   ```
   

### Command line options

To get a complete list of all command-line arguments with descriptions and default values you can run:

```
python main.py --help
```

### Dataset

The COCO2017 training dataset contains 118’287 images. Hence the number of steps per epoch for different batchsizes are:

| Batch Size | Iterations per Epoch |
| ---------- | -------------------- |
| 32         | 3696                 |
| 64         | 1848                 |
| 128        | 924                  |
| 1024       | 115                  |
| 2048       | 57                   |

### Goal

### Performance

| Cores | Batchsize/Core | Accumulation | BF16 | Throughput | Epoch Time | Tot. Time |
| ----- | -------------- | ------------ | ---- | ---------- | ---------- | --------- |
| 1     | 128            | -            | -    | 106        | 18.5 min   | 21h       |
| 1     | 32             | -            | -    | 89         | -          | -         |
|       |                |              |      |            |            |           |

