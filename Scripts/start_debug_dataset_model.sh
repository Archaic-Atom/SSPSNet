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
CUDA_VISIBLE_DEVICES=1 nohup python -u Source/main.py \
                        --mode train \
                        --batchSize 4 \
                        --gpu 2 \
                        --trainListPath ./Datasets/sceneflow_stereo_training_list.csv \
                        --valListPath ./Datasets/sceneflow_stereo_val_list.csv \
                        --imgWidth 448 \
                        --imgHeight 224 \
                        --dataloaderNum 24 \
                        --maxEpochs 200 \
                        --imgNum 35454 \
                        --valImgNum 0 \
                        --sampleNum 1 \
                        --log ${tensorboard_folder} \
                        --outputDir ./DebugResult/ \
                        --lr 0.00015 \
                        --dist False \
                        --modelName FANet \
                        --port ${dist_port} \
                        --modelDir ./Checkpoint_debug/ \
                        --debug False \
                        --auto_save_num 1 \
                        --dataset sceneflow > DebugRun.log 2>&1 &
echo "You can use the command (>> tail -f TrainRun.log) to watch the training process!"

echo "Start the tensorboard at port:" ${tensorboard_port}
nohup tensorboard --logdir ${tensorboard_folder} --port ${tensorboard_port} \
                        --bind_all --load_fast=false > Tensorboard.log 2>&1 &
echo "All processes have started!"

echo "Begin to watch TrainRun.log file!"
tail -f DebugRun.log
