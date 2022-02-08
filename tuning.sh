'''
python tuning_DST.py \
  --seed 100 --method direct \
  --do_tensorize --n_gpu 1 --n_process 40
'''

python tuning_DST.py \
  --task DST --k 1 --test_k 1 --seed 100 --train_seed 1 --method direct --n_gpu 1 \
  --batch_size 1 --lr 1e-05 --fp16 --optimization 8bit-adam --out_dir checkpoints/DST/
