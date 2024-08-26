#!/bin/bash

#sbatch /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/bash/feature_extractor/extract_features_alexnet.sh
#sbatch /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/bash/feature_extractor/extract_features_inception_v3.sh
sbatch /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/bash/feature_extractor/extract_features_resnet50.sh
sbatch /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/bash/feature_extractor/extract_features_resnet101.sh
# sbatch ./extract_features_vgg16.sh
#sbatch /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/bash/feature_extractor/extract_features_vgg19.sh
#sbatch /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/bash/feature_extractor/extract_features_efficient_v2_s.sh
# sbatch /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/bash/feature_extractor/extract_features_vit_b_32.sh
# sbatch /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/bash/feature_extractor/extract_features_vit_b_16.sh
#sbatch /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/bash/feature_extractor/extract_features_resnet18.sh