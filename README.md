# Background Aware Metric Learning

# Training
```bash
 python train.py --base inception_v1 --batch_size 32
```
Checkout more command line options in `train.py` file. `dnn_library.py` is the interface to use any other base feature extractor.

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
