# A script to generate noisy files 
for parameters see noise_config.yaml

- the script will generate noisy files from noise files and clean files and create  noisy_records.txt which can be split for test and train
- noise is added with different  snr values (snr_min, snr_max)
- noise and clean files should be provided (see noise.txt, clean_files.txt)
- white_noise: white noise is added randomly with some probability
- num_copies:  each clean file mixed with n noisy files resulting in a product
- if noise file is shorter than clean (speech file) it will be copied several times