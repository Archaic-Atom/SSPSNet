#!/bin/bash
# parameters
tensorboard_port=6237
dist_port=8819
tensorboard_folder='./log/'
log_file='TrainRun_debug.log'
echo "The tensorboard_port:" ${tensorboard_port}
echo "The dist_port:" ${dist_port}

training_list='./Datasets/kitti2012_stereo_training_list.csv'

# command
# delete the previous tensorboard files
if [ -d "${tensorboard_folder}" ]; then
    rm -r ${tensorboard_folder}
fi


echo "Begin to train the model!"
CUDA_VISIBLE_DEVICES=0,1,2,3,6,7 nohup python -u Source/main.py \
                        --mode train \
                        --batchSize 3 \
                        --gpu 6 \
                        --trainListPath ${training_list}\
                        --valListPath ./Datasets/sceneflow_stereo_val_list.csv \
                        --imgWidth 518 \
                        --imgHeight 266 \
                        --dataloaderNum 4 \
                        --maxEpochs 200 \
                        --imgNum 35454 \
                        --valImgNum 0 \
                        --sampleNum 1 \
                        --log ${tensorboard_folder} \
                        --lr 0.0001 \
                        --dist TRUE \
                        --modelName StereoA \
                        --port ${dist_port} \
                        --modelDir ./Checkpoint_kitti2012/ \
                        --debug False \
                        --auto_save_num 20 \
                        --dataset kitti2012 > ${log_file} 2>&1 &
echo "You can use the command (>> tail -f ${log_file}) to watch the training process!"

echo "Start the tensorboard at port:" ${tensorboard_port}
nohup tensorboard --logdir ${tensorboard_folder} --port ${tensorboard_port} \
                        --bind_all --load_fast=false > Tensorboard.log 2>&1 &
echo "All processes have started!"

echo "Begin to watch TrainRun.log file!"
tail -f ${log_file}
