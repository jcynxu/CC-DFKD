python datafree_kd.py \
--method CC-DFKD \
--dataset cifar100 \
--batch_size 256 \
--synthesis_batch_size 256 \
--teacher wrn40_2 \
--student wrn16_2 \
--lr 0.08 \
--kd_steps 400 \
--ep_steps 400 \
--g_steps 200 \
--lr_g 2e-3 \
--gpu 1 \
--seed 0 \
--T 20 \
--save_dir run/sample_CC-DFKD \
--log_tag log_CC-DFKD \
--co_alpha 5 \
--co_beta 1 \
--co_gamma 0.7 \
--co_eta 0.7