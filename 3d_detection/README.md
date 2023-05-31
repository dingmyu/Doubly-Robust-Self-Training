## Getting Started

Please refer to [getting_started.md](docs/en/getting_started.md) for installation.


## Training
To train the student model with the combined labels + pseudo_labels, run the following command:
```
./tools/dist_train.sh configs/centerpoint/centerpoint_doubly_robust_fourth_ft.py 3
```

We provide separate config files for both the naive autolabeling loss and doubly-robust loss, as well as for several fractions of labeled data. The pre-trained teacher weights used to generate the pseudo-labels can be modified in the config file.

To run inference on the validation set, run the following command:

```
python tools/test.py configs/centerpoint/centerpoint_doubly_robust_fourth_ft.py work_dirs/centerpoint_doubly_robust_fourth_ft/epoch_10.pth --eval mAP
```