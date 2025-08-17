#!/bin/bash
# parameter
test_gpus_id=4
eva_gpus_id=0
test_list_path='./Datasets/kitti2012_stereo_training_list.csv'
evalution_format='training'
start_epoch=45
epoch_num=46
result_path=./Result/test_output_kitti2012.txt
model_dir=./checkpoint_l/

rm -r ${result_path}
for((i=${start_epoch}; i<${start_epoch}+${epoch_num}; i++));
do
    echo "test gpus id: "${test_gpus_id}
    echo "the list path is: "${test_list_path}
    echo "start to predict disparity map"
    model_path="${model_dir}model_epoch_${i}.pth"
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
                            --modelName StereoC \
                            --outputDir ./TestResult/ \
                            --modelDir ${model_path} \
                            --resultImgDir ./resultImg_kitti12/ \
                            --dataset kitti2015
    echo "Finish!"

    CUDA_VISIBLE_DEVICES=${test_gpus_id} python Source/Tools/evalution_stereo_net.py \
     --gt_list_path ${test_list_path} \
     --epoch ${i} \
     --output_path ${result_path} \
     --img_path_format './resultImg_kitti12/%06d_10.png'
    sleep 10
done
echo "Finish!"