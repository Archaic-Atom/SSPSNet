#!/bin/bash
test_gpus_id=0
test_list_path='./Datasets/kitti2015_stereo_testing_list.csv'
model_dir=./Checkpoint_kitti2012/

rm -r ResultImg
CUDA_VISIBLE_DEVICES=${test_gpus_id} python -u Source/main.py \
                            --mode test \
                            --batchSize 1 \
                            --gpu 1 \
                            --trainListPath ${test_list_path} \
                            --imgWidth 1 \
                            --imgHeight 1 \
                            --dataloaderNum 4 \
                            --maxEpochs 1 \
                            --imgNum 200 \
                            --sampleNum 1 \
                            --lr 0.0001 \
                            --log ./TestLog/ \
                            --dist False \
                            --modelName StereoA \
                            --outputDir ./TestResult/ \
                            --modelDir ${model_dir} \
                            --resultImgDir ./ResultImg/ \
                            --dataset kitti2015

mv ResultImg disp_0
zip -r disp_0.zip disp_0
mv disp_0.zip ./Submission/
mv disp_0 ResultImg