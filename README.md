# Code for Doubly-Robust Self-Training (the Image Classification Part)

### Getting Started
Python3, PyTorch>=1.8.0, torchvision>=0.7.0 are required for the current codebase.

```shell
# An example on CUDA 10.2
pip install torch===1.9.0+cu102 torchvision===0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install thop pyyaml fvcore pillow==8.3.2
```

Other pytorch or CUDA versions should also work, just make sure they are installed correctly!

### Dataset
- Prepare the ImageNet dataset in the timm format
```shell
- DATASET_DIR/
    - train/
        - ClassFolder1
        - ClassFolder2
        - ...
    - val/
        - ClassFolder1
        - ClassFolder2
        - ...
```  

**In this project, the image pseudo label is read directly from the dataset folder (e.g. folder ClassFolder1 means class 1), and the ground truth label is read from the image name (e.g., ClassFolder1/groundtruth_Class2_image1.png means the ground truth belongs to Class2).**

### Training and Validation
- Set the following ENV variable:
  ```
  $MASTER_ADDR: IP address of the node 0 (Not required if you have only one node (machine))
  $MASTER_PORT: Port used for initializing distributed environment
  $NODE_RANK: Index of the node
  $N_NODES: Number of nodes 
  $NPROC_PER_NODE: Number of GPUs (NOTE: should exactly match local GPU numbers with `CUDA_VISIBLE_DEVICES`)
  ```

- Training:
  - Example1 (One machine with 8 GPUs):
  ```shell
  python -u -m torch.distributed.launch --nproc_per_node=8 \
  --nnodes=1 --node_rank=0 --master_port=12345 \
  train.py DATASET_DIR --model DaViT_tiny --batch-size 128 --lr 1e-3 \
  --native-amp --clip-grad 1.0 --output OUTPUT_DIR --num-classes 100 --epochs 20 --mixup 0 --cutmix 0 --pseudomode 0
  ```

  - Example2 (Two machines, each has 8 GPUs):
  ```shell
  # Node 1: (IP: 192.168.1.1, and has a free port: 12345)
  python -u -m torch.distributed.launch --nproc_per_node=8
  --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
  --master_port=12345 train.py DATASET_DIR --model DaViT_tiny --batch-size 128 --lr 2e-3 \
  --native-amp --clip-grad 1.0 --output OUTPUT_DIR  --num-classes 100 --epochs 20 --mixup 0 --cutmix 0 --pseudomode 0

  # Node 2:
  python -u -m torch.distributed.launch --nproc_per_node=8
  --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
  --master_port=12345 train.py DATASET_DIR --model DaViT_tiny --batch-size 128 --lr 2e-3 \
  --native-amp --clip-grad 1.0 --output OUTPUT_DIR  --num-classes 100 --epochs 20 --mixup 0 --cutmix 0 --pseudomode 0
  ```

**Note that the pseudomode 0 means using the labels according the data folder (Pseudo Only), pseudomode 1 means Pseudo + Labeled, and pseudomode 2 means Doubly Robust.**


- Validation:
  ```shell
  CUDA_VISIBLE_DEVICES=0 python -u validate.py DATASET_DIR --model DaViT_tiny --batch-size 128  \
  --native-amp  --checkpoint TRAINED_MODEL_PATH
  ```

- Get Pseudo Labels (then create symlink according to the predicted pseudo labels for doubly robust training):
  ```shell
  python -u inference.py DATASET_DIR/train/ --model DaViT_tiny --batch-size 1024 \
  --checkpoint checkpoint.pth.tar \
  --num-gpu 8 --num-classes 100
  ```
