# <a href="http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf"> Deep Adversarial Metric Learning </a> Paper implementation

# Training
```bash
 python train.py --base inception_v1 --batch_size 32
```

Command Line options:
* `--base` : Base architecture of the feature extractor (Default : inception_v1)
* `--checkpoint` : checkpoint to load (Initially load the feature extractor checkpoint. Once trained load the combined model checkpoint)
* `--metric_weight` : weight for J_m
* `--reg_weight` : weight for J_reg
* `--adv_weight` : weight for J_adv

dnn_library.py is the interface to use any other base feature extractor.

# Generate Data
In the data folder, you can find scripts for generating positive pairs and tf records needed for training


## To generate text file of positive pairs
```bash
python gen_pos_pairs.py
```
This will generate two files train_pos_pairs.txt and test_pos_pairs.txt

## To generate TF records
```bash
python gen_data.py --path <path_to_train_pos_pairs.txt> -n <Number of examples in TF record> 
```
* `--path` : Path to train_pos_pairs.txt
* `--n` : Number of examples in  TF record (Default: It will write all the examples in train_pos_pairs.txt)

# Checkpoints and summaries can be found at 

```bash 
/shared/kgcoe-research/mil/peri/birds_data/experiments
```