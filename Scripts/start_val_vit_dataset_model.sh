#!/bin/bash
# parameters
tensorboard_port=6234
dist_port=8809
tensorboard_folder='./log/'
echo "The tensorboard_port:" ${tensorboard_port}
echo "The dist_port:" ${dist_port}

# command
# delete the previous tensorboard files
if [ -d "${tensorboard_folder}" ]; then
    rm -r ${tensorboard_folder}
fi

echo "Begin to train the model!"
CUDA_VISIBLE_DEVICES=0 nohup python -u Source/main.py \
                        --mode train \
                        --batchSize 4 \
                        --gpu 1 \
                        --trainListPath ./Datasets/sceneflow_stereo_training_list.csv \
                        --valListPath ./Datasets/sceneflow_stereo_val_list.csv \
                        --imgWidth 980 \
                        --imgHeight 560 \
                        --dataloaderNum 24 \
                        --maxEpochs 1 \
                        --imgNum 0 \
                        --valImgNum 35454 \
                        --sampleNum 1 \
                        --log ${tensorboard_folder} \
                        --lr 0.001 \
                        --dist False \
                        --modelName FANet \
                        --port ${dist_port} \
                        --modelDir ./Checkpoint_Test/ \
                        --debug False \
                        --auto_save_num 5 \
                        --dataset sceneflow > TrainRun.log 2>&1 &
echo "You can use the command (>> tail -f TrainRun.log) to watch the training process!"

echo "Start the tensorboard at port:" ${tensorboard_port}
nohup tensorboard --logdir ${tensorboard_folder} --port ${tensorboard_port} \
                        --bind_all --load_fast=false > Tensorboard.log 2>&1 &
echo "All processes have started!"

echo "Begin to watch TrainRun.log file!"
tail -f TrainRun.log
