#!/bin/bash
# parameter
test_gpus_id=7
eva_gpus_id=0
test_list_path='./Datasets/sceneflow_stereo_testing_list.csv'
evalution_format='training'

echo "test gpus id: "${test_gpus_id}
echo "the list path is: "${test_list_path}
echo "start to predict disparity map"
CUDA_VISIBLE_DEVICES=${test_gpus_id} python -u Source/main.py \
                        --mode web \
                        --batchSize 1 \
                        --gpu 1 \
                        --trainListPath ${test_list_path} \
                        --imgWidth 1288 \
                        --imgHeight 840 \
                        --dataloaderNum 4 \
                        --maxEpochs 1 \
                        --imgNum 200 \
                        --sampleNum 1 \
                        --lr 0.0001 \
                        --log ./TestLog/ \
                        --dist False \
                        --modelName StereoA \
                        --outputDir ./TestResult/ \
                        --modelDir ./pre_trained/ \
                        --dataset sceneflow
echo "Finish!"