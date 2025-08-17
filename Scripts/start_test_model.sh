#!/bin/bash
# parameter
test_gpus_id=1
eva_gpus_id=0
# test_list_path='./Datasets/eth3d_stereo_training_list.csv'
# test_list_path='./Datasets/kitti2012_stereo_training_list.csv'
test_list_path='./Datasets/kitti2015_stereo_training_list.csv'
# test_list_path='./Datasets/sceneflow_stereo_testing_list.csv'
# test_list_path='./Datasets/cretereo_training_list.csv'
# test_list_path='./Datasets/cretereo_training_list.csv'
# test_list_path='./Datasets/kitti2012_stereo_testing_list.csv'
evalution_format='training'
# confidence_level=0.08
confidence_level=0.15
# confidence_level=0.12
# confidence_level=0.17
result_path=./Result/test_output_mask_eth3d.txt

echo "test gpus id: "${test_gpus_id}
echo "the list path is: "${test_list_path}
echo "the confidence_level is: "${confidence_level}
echo "start to predict disparity map"

rm -r ./ResultImg
CUDA_VISIBLE_DEVICES=${test_gpus_id} python -u Source/main.py \
                        --mode test \
                        --batchSize 1 \
                        --gpu 1 \
                        --trainListPath ${test_list_path} \
                        --imgWidth 10 \
                        --imgHeight 10 \
                        --dataloaderNum 4 \
                        --maxEpochs 1 \
                        --imgNum 200 \
                        --sampleNum 1 \
                        --lr 0.0001 \
                        --log ./TestLog/ \
                        --dist False \
                        --modelName StereoA \
                        --outputDir ./TestResult/ \
                        --modelDir ./Checkpoint_two/ \
                        --confidence_level ${confidence_level} \
                        --dataset kitti2012
echo "Finish!"

CUDA_VISIBLE_DEVICES=${test_gpus_id} python Source/Tools/evalution_stereo_net.py \
     --gt_list_path ${test_list_path} \
     --epoch 1 \
     --output_path ${result_path} \
     --img_path_format './ResultImg/%06d_10.png'