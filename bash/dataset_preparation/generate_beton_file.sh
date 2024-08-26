#!/bin/bash


write_dataset () {
    write_path=/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/data/${1}_${2}_${3}.ffcv
    echo "Writing ${1} dataset split ${2} to ${write_path} with max resolution ${3}, compress probability ${4}, and jpeg quality ${5}"
    python /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/scripts/generate_beton_file.py \
        --cfg.dataset=${1} \
        --cfg.split=${2} \
        --cfg.write_path=$write_path \
        --cfg.max_resolution=${3} \
        --cfg.write_mode=proportion \
        --cfg.compress_probability=${4} \
        --cfg.jpeg_quality=${5}
}

write_dataset lamem train 299 0.5 90 # dataset_name, split, max_resolution, compress_probability, jpeg_quality
write_dataset lamem validation 299 0.5 90 # dataset_name, split, max_resolution, compress_probability, jpeg_quality
write_dataset lamem test 299 0.5 90 # dataset_name, split, max_resolution, compress_probability, jpeg_quality
# write_dataset imagenet validation 256 0.5 90 # dataset_name, split, max_resolution, compress_probability, jpeg_quality
# write_dataset imagenet train 256 0.5 90 # dataset_name, split, max_resolution, compress_probability, jpeg_quality
# write_dataset imagenet test 256 0.5 90 # dataset_name, split, max_resolution, compress_probability, jpeg_quality




# write_dataset $1 $2 $3 $4 $5 # dataset_name, split, max_resolution, compress_probability, jpeg_quality


# Example
# ./bash/generate_beton_file.sh imagenet train 256 0.5 90
# ./bash/generate_beton_file.sh imagenet validation 256 0.5 90
# ./bash/generate_beton_file.sh lamem train 256 0.5 90
# ./bash/generate_beton_file.sh lamem validation 256 0.5 90
# ./bash/generate_beton_file.sh lamem test 256 0.5 90
