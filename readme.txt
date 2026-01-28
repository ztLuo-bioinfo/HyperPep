python main.py --load_ckpt checkpoints/hyperpep_fg.pt --test_name test_zero-shot.csv

python main.py --load_ckpt checkpoints/hyperpep_fg.pt --test_name test_covid19.csv

python main.py --load_ckpt checkpoints/hyperpep_pg_k3.pt --test_name test_tcr_split.csv

python main.py --load_ckpt checkpoints/hyperpep_pg_k3.pt --test_name test_strict_split.csv