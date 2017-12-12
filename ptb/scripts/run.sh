for seed in 2 1 3
do
  python PTB-release.py --seed=${seed} --opt_method=YF --log_dir=../results/YF_seed_${seed}_h_max_log_test_slow_start_10_win_h_max_clip --h_max_log_smooth
done

