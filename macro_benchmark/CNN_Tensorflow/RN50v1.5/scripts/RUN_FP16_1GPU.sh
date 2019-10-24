mpiexec --allow-run-as-root --bind-to socket -np 1 /usr/bin/python main.py --mode=training_benchmark --batch_size=256 --warmup_steps=200 --num_iter=400 --precision=fp32 --iter_unit=batch --data_dir=/data//source_data/build_imagenet_data-rebuild/ --results_dir=/tmp/log  --use_tf_amp  
