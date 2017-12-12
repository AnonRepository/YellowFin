for seed in 2 1 3 
do
  mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=../results/YF_seed_${seed}_h_max_log_test_slow_start_10_win_h_max_clip --opt_method="YF" --seed=${seed} --h_max_log_smooth=1
done

