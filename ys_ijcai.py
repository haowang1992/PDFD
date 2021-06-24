import os
import argparse

parser = argparse.ArgumentParser('baseline loss weight grid search')
parser.add_argument('--dataset', type=str, choices=['Sketchy', 'TU-Berlin'])
parser.add_argument('--test', action='store_true', default=False, help='test')
opt = parser.parse_args()

if opt.dataset == 'Sketchy':
	if not opt.test:
		cmd = f'PYTHONPATH=`pwd` python train_ijcai.py --dataset Sketchy_extended ' \
			f'--dim_out 64 --semantic_models word2vec-google-news hieremb-jcn ' \
			f'--dataset_root ./ZS-SBIR ' \
			f'--epochs 20 --early_stop 5 --lr 0.0001 --gpu_id 0 --seed 0 ' \
			f'--lambda_ret_cls {0.1/10.0} --lambda_domain_cls {10.0/10.0} --lambda_rec {10.0/10.0} --drop {5.0/10.0}'
		os.system(cmd)
	else:
		cmd = f'PYTHONPATH=`pwd` python train_ijcai.py --dataset Sketchy_extended ' \
			  f'--dim_out 64 --semantic_models word2vec-google-news hieremb-jcn ' \
			  f'--dataset_root ./ZS-SBIR --test ' \
			  f'--epochs 20 --early_stop 5 --lr 0.0001 --gpu_id 0 --seed 0 ' \
			  f'--lambda_ret_cls {0.1 / 10.0} --lambda_domain_cls {10.0 / 10.0} --lambda_rec {10.0 / 10.0} --drop {5.0 / 10.0}'
		os.system(cmd)

elif opt.dataset == 'TU-Berlin':
	if not opt.test:
		cmd = f'PYTHONPATH=`pwd` python train_ijcai.py --dataset TU-Berlin ' \
			f'--dim_out 64 --semantic_models glove-wiki-gigaword hieremb-jcn ' \
			f'--dataset_root ./ZS-SBIR ' \
			f'--epochs 20 --early_stop 5 --lr 0.0001 --gpu_id 0 --seed 0 ' \
			f'--lambda_ret_cls {4.0/10.0} --lambda_domain_cls {4.0/10.0} --lambda_rec {5.0/10.0} --drop {5.0/10.0}'
		os.system(cmd)
	else:
		cmd = f'PYTHONPATH=`pwd` python train_ijcai.py --dataset TU-Berlin ' \
			  f'--dim_out 64 --semantic_models glove-wiki-gigaword hieremb-jcn ' \
			  f'--dataset_root ./ZS-SBIR --test ' \
			  f'--epochs 20 --early_stop 5 --lr 0.0001 --gpu_id 0 --seed 0 ' \
			  f'--lambda_ret_cls {4.0 / 10.0} --lambda_domain_cls {4.0 / 10.0} --lambda_rec {5.0 / 10.0} --drop {5.0 / 10.0}'
		os.system(cmd)



