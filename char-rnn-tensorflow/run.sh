for seed in 2 1 3
do
  python train_YF.py --log_dir=./results/YF_seed_${seed}_h_max_log_test_slow_start_10_win_h_max_clip --data_dir=./data/tinyshakespeare/ --opt_method=YF --seed=${seed} --h_max_log_smooth
done
