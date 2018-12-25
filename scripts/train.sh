#/bin/bash

# Define the experiment configuration
batch_size=128
num_epochs=450
base='inception_v1'  # Base architecture of CNN feature extractor
checkpoint='/shared/kgcoe-research/mil/peri/tf_checkpoints/inception_v1.ckpt' # CMR model checkpoint for finetuning mode
# checkpoint='/shared/kgcoe-research/mil/peri/birds_data/experiments/c_triplet_single_scratch_2018-11-27_12_04/model.ckpt-2000' # CMR model checkpoint for finetuning mode
embedding_dim=512  # CVS dimension
lr=0.0001
decay_steps=15000
decay_factor=0.96
save_steps=2000 # Step interval for checkpoint saving
margin=1.0 # Margin for metric loss
record_path='/shared/kgcoe-research/mil/peri/birds_data/birds_ob_train_mask.tfrecord' # TFRecord path to read from
model='object_whole_separate'
mode='scratch'
exp_path='/shared/kgcoe-research/mil/peri/birds_data/experiments'
optimizer='adam'

python ./train.py --batch_size ${batch_size} \
                  --num_epochs ${num_epochs} \
                  --save_steps ${save_steps} \
                  --base ${base} \
                  --embedding_dim ${embedding_dim} \
                  --lr ${lr} \
                  --margin ${margin} \
                  --decay_steps ${decay_steps} \
                  --decay_factor ${decay_factor} \
                  --record_path ${record_path} \
                  --checkpoint ${checkpoint} \
                  --optimizer ${optimizer} \
                  --exp_path ${exp_path} \
                  --model ${model} \
                  --mode ${mode}
                   
                           
                           
                           
                           
                           
                           
                          
                           
