# The implementation of Beacon in the IJCAI'19 paper named "Correlation-Sensitive Next-Basket Recommendation""


@Input format(s): Train/Validate/Test sets have the same format

For each basket sequence, baskets {b_i} are separated by '|'
e.g., b_1|b_2|b_3|...|b_n
For each basket b_i, items {v_j} are separated by a space ' '
e.g., v_1 v_2 v_3 ... v_m

@How to train:
 i) Step 1: Generate pre-computed correlation matrix C using cmatrix_generator.py. 
     + The 'nbhop' parameter is to generate the Nth-order correlation matrix
     + The default output directory is "data_dir/adj_matrix"
 ii) Step 2: Train the Beacon model using main_gpu.py ".
     + Support 3 modes: train_mode, prediction_mode, tune_mode
     + The format of the prediction file is as follows: 
       Target:gt_basket|item_candidate_1:score_1|item_candidate_2:score_2|

@Please drop me an email (ductrong.le.2014 at smu.edu.sg) if you need any clarification. Thanks
