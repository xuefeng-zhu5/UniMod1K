# UniMod1K: Towards a More Universal Large-Scale Dataset and Benchmark for Multi-Modal Learning

The official implementation of the multi-modal (Vision, Depth and Language) SPT tracker of the paper **UniMod1K: Towards a More Universal Large-Scale Dataset and Benchmark for Multi-Modal Learning**

<center><img width="75%" alt="" src="./spt_vdl_framework.jpg"/></center>

## Usage
### Installation

Install the environment using Anaconda
```
conda create -n spt python=3.6
conda activate spt
bash install_pytorch17.sh
cd /path/to/UniMod1K/SPT
```

### Data Preparation
The training dataset is the [**UniMod1K**]
```
--UniMod1K
    |--Adapter
        |--adapter1
        |--adapter2
        ...
    |--Animal
       |--alpaca1
       |--bear1
        ...
    ... 
```

### Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```

After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```
### Training
Download the pretrained weight [[BERT pretrained weight](https://drive.google.com/drive/folders/1Fi-4TSaIP4B_TPi2Jme2sxZRdH9l5NPN?usp=share_link)] put it under `$PROJECT_ROOT$/pretrained_models`. 
Set the MODEL.LANGUAGE.PATH and MODEL.LANGUAGE.VOCAB_PATH in ./experiments/spt/unimod1k.yaml.

Download the pretrained [Stark-s model](https://drive.google.com/drive/folders/142sMjoT5wT6CuRiFT5LLejgr7VLKmaC4)
and put it under `$PROJECT_ROOT$/pretrained_models/`. 
Set the MODEL.PRETRAINED path in ./experiments/spt/unimod1k.yaml.

Training with multiple GPUs using DDP (4 RTX3090Ti with batch size of 16)
```
export PYTHONPATH=/path/to/SPT:$PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=4 ./lib/train/run_training.py  
```
or using single GPU:
```
python ./lib/train/run_training.py  
```

### Test
Edit ./lib/test/evaluation/local.py to set the test set path, then run
```
python ./tracking/test.py
```
You can also use the [pre-trained model](https://drive.google.com/file/d/1aU1FWERBab0aGR9nxwN138JG1lLQlnU5/view?usp=drive_link), 
and set the path in ./lib/test/parameter/spt.py

### Evaluation
Put the raw results in the [VOT Toolkit](https://github.com/votchallenge/toolkit) workspace, then use the command of vot analysis. The tutorial of VOT Toolkit can be found [here](https://www.votchallenge.net/howto/overview.html).

## Acknowledgment
- This repo is based on [Stark](https://github.com/researchmm/Stark) which is an excellent work.

## Contact
If you have any question, please feel free to [contact us](xuefeng_zhu95@163.com)(xuefeng_zhu95@163.com)