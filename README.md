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

- The [Nvidia repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD#setup)  reports final accuracies of $\approx 0.25$.
  I believe that what they report is `mAP@[0.5:0.95]`(see [mAP explanation](https://datascience.stackexchange.com/questions/16797/what-does-the-notation-map-5-95-mean)).
- This would kind of match with the results in the original [SSD paper](https://arxiv.org/pdf/1512.02325.pdf), where they report 23.2 mAP with a different dataset (COCO2015 instead of COCO2017) and probably a slighly different training procedure.
- The Nvidia repository shows the following loss curves:

![training_loss](PyTorch/Detection/SSD/img/training_loss.png)

### Performance

Using DALI for preprocessing, our workload is preprocessing/CPU bound, hence scaling to 8 cores results in degraded performance.

| Cores | Batchsize/Core   | Acc. | BF16 | Throughput | Epoch Time | Tot. Time |
| ----- | ---------------- | ---- | ---- | ---------- | ---------- | --------- |
| 1     | 128              | -    | -    | 145        | 13.5       | 16h       |
| 1     | 128              | -    | YES  | 225        | -          | -         |
| 1     | 256              | -    | YES  | 200        | -          | -         |
| 1     | 128 (–fake_data) | -    | YES  | 230        | -          | -         |
| 8     | 128              | -    | YES  |            | -          | -         |
| 8     | 128 (–fake_data) | -    | YES  | 200        | -          | -         |
| 8     | 128 (–fake_data) | 10   | YES  |            | -          | -         |

