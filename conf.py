import os

raw_data_dir = '/home/aleks/data/speech'
snodgrass_data_dir = '/home/aleks/data/speech/Naming_Data_Complete/Snodgrass_Recordings_cleaned'
processed_data_dir = '/home/aleks/data/speech_processed'
res_dir = '/home/aleks/projects/thesis/res'
awe_runs_dir = '/home/aleks/projects/thesis/acoustic_word_embeddings/runs'

# the final patient + healthy dataset
current_dataset = 'all_snodgrass_cleaned_v5'

# the clean dataset (used in section 4.3.4 of the thesis)
# current_dataset = 'independent_cleaned_v3'

# a dataset with previously unseen words, not used anymore
new_path = os.path.join(processed_data_dir, 'all_new_words.scp')
