#!/usr/bin/env bash

source ~/venv/thesis/bin/activate
export PYTHONPATH=~/projects/thesis/


python rating_system.py --run_dir /home/aleks/projects/thesis/acoustic_word_embeddings/runs_cluster/siamese_43_20_12_2018 --run_epoch 51 --ratings_file /home/aleks/data/speech_processed/all_snodgrass_cleaned_v5_test_ratings_full --vad
python rating_system.py --run_dir /home/aleks/projects/thesis/acoustic_word_embeddings/runs_cluster/siamese_53_20_12_2018 --run_epoch 66 --ratings_file /home/aleks/data/speech_processed/all_snodgrass_cleaned_v5_test_ratings_full --vad

# non-aug
#python rating_system.py --run_dir /home/aleks/projects/thesis/cluster_logs/2019/runs_standard_plus_noise_2019/siamese_0_29_01_2019 --run_epoch 58 --ratings_file /home/aleks/data/speech_processed/all_snodgrass_cleaned_v5_test_ratings_full --vad
# aug
#python rating_system.py --run_dir /home/aleks/projects/thesis/acoustic_word_embeddings/runs_cluster/siamese_53_20_12_2018 --run_epoch 66 --ratings_file /home/aleks/data/speech_processed/all_snodgrass_cleaned_v5_test_ratings_full --vad