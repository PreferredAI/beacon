# beacon
The implementation of "Correlation-Sensitive Next-Basket Recommendation"", published in IJCAI'19


@Input format(s):
For each basket sequence, baskets {b_i} are separated by '|'
e.g., b_1|b_2|b_3|...|b_n
For each basket b_i, items {v_j} are separated by a space ' '
e.g., v_1 v_2 v_3 ... v_m

@How to run: main_gpu.sh

Use --train_mode to enable the training mode
Use --prediction_mode to generate evaluation metrics

@How to collect results from different seeds: Use collect_result.sh
