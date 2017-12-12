for seed in 2 1 3
do
  python CIFAR10-release.py --seed=${seed} --log_dir=../results/YF_seed_${seed}_h_max_log_test_slow_start_10_win_h_max_clip --opt_method=YF --h_max_log_smooth
done
