## The implementation of Beacon in the "[Correlation-Sensitive Next-Basket Recommendation](https://www.ijcai.org/proceedings/2019/0389.pdf)" paper (IJCAI'19)


1. Input format(s): Train/Validate/Test sets have the same format

 - For each basket sequence, baskets {b_i} are separated by '|', e.g., b_1|b_2|b_3|...|b_n

 - For each basket b_i, items {v_j} are separated by a space ' ', e.g., v_1 v_2 v_3 ... v_m

2. How to train:
 - Step 1: Generate pre-computed correlation matrix C using cmatrix_generator.py. 
     + The 'nbhop' parameter is to generate the Nth-order correlation matrix
     + The default output directory is "data_dir/adj_matrix"
 - Step 2: Train the Beacon model using main_gpu.py ".
     + Support 3 modes: train_mode, prediction_mode, tune_mode
     + The format of the prediction file is as follows: 
       Target:gt_basket|item_candidate_1:score_1|item_candidate_2:score_2|

3. If you find the code useful in your research, please cite:

```
@inproceedings{le2019beacon,
  title={Correlation-Sensitive Next-Basket Recommendation},
  author={Le, Duc-Trong, Lauw, Hady W and Fang, Yuan},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence},
  year={2019},
}
```

## Requirements

- Python == 3.6
- Tensorflow == 1.14
- scipy.sparse == 1.3.0

@Please drop me an email (ductrong.le.2014 at smu.edu.sg) if you need any clarification. Thanks :+1:
