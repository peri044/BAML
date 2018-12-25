#/bin/bash

# Define the experiment configuration
batch_size=20
num_epochs=1
num=5900
base='inception_v1'
checkpoint=$1
embedding_dim=512
model='object_whole_separate'
mode='val'
record_path='/shared/kgcoe-research/mil/peri/birds_data/birds_ob_test_mask.tfrecord'

python ./eval.py --batch_size ${batch_size} \
                 --num_epochs ${num_epochs} \
                 --base ${base} \
                 --checkpoint ${checkpoint} \
                 --embedding_dim ${embedding_dim} \
                 --record_path ${record_path} \
                 --model ${model} \
                 --mode ${mode} \
                 --num ${num}
                 
                  
                           