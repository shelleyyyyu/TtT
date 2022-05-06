suffix="baseline"
bpath="./model/bert/"
cpath="/data/yumenghsuan/cgec/ckpt/NLPCC_"$suffix"/"

mkdir -p $cpath

nohup python -u main.py \
    --bert_path  $bpath/bert.ckpt\
    --bert_vocab $bpath/vocab.txt \
    --train_data ./data/NLPCC/nlpcc/train.txt \
    --dev_data ./data/NLPCC/nlpcc/test.txt\
    --test_data ./data/NLPCC/nlpcc/test.txt\
    --batch_size 100 \
    --lr 1e-5 \
    --dropout 0.1 \
    --number_epoch 30 \
    --gpu_id 0 \
    --print_every 50 \
    --save_every 500 \
    --fine_tune \
    --loss_type FC_FT_CRF \
    --gamma 0.5 \
    --model_save_path $cpath \
    --prediction_max_len 64 \
    --dev_eval_path $cpath/dev_pred.txt \
    --final_eval_path $cpath/dev_eval.txt \
    --test_eval_path $cpath/test_eval_%d.txt \
    --l2_lambda 1e-5 \
    --training_max_len 64 > ./logs/train.nlpcc.baseline.log 2>&1&

########################################################################

suffix="baseline"
bpath="./model/bert/"
cpath="/data/yumenghsuan/cgec/ckpt/NLPCC_"$suffix"/"

mkdir -p $cpath

nohup python -u evaluation_test.py \
    --bert_path  $bpath/bert.ckpt\
    --bert_vocab $bpath/vocab.txt \
    --train_data ./data/NLPCC/nlpcc/train.txt \
    --dev_data ./data/NLPCC/nlpcc/test.txt\
    --test_data ./data/NLPCC/nlpcc/test_paraphrasing.txt\
    --batch_size 100 \
    --lr 1e-5 \
    --dropout 0.1 \
    --number_epoch 30 \
    --gpu_id 0 \
    --print_every 50 \
    --save_every 500 \
    --fine_tune \
    --loss_type FC_FT_CRF \
    --gamma 0.5 \
    --model_save_path $cpath \
    --prediction_max_len 64 \
    --dev_eval_path $cpath/dev_pred.txt \
    --final_eval_path $cpath/dev_eval.txt \
    --test_eval_path $cpath/test_eval_%d.txt \
    --l2_lambda 1e-5 \
    --training_max_len 64 \
    --restore_ckpt_path $cpath/epoch_29_dev_f1_0.460 > ./logs/test.nlpcc.baseline.log 2>&1&




####### DEBUG
python -u main.py \
    --bert_path  ./model/bert/bert.ckpt\
    --bert_vocab ./model/bert/vocab.txt \
    --train_data ./data/NLPCC/nlpcc/train_debug.txt \
    --dev_data ./data/NLPCC/nlpcc/test_debug.txt\
    --test_data ./data/NLPCC/nlpcc/test_debug.txt\
    --batch_size 10 \
    --lr 1e-5 \
    --dropout 0.1 \
    --number_epoch 1 \
    --gpu_id 0 \
    --print_every 1 \
    --save_every 1 \
    --fine_tune \
    --loss_type FC_FT_CRF \
    --gamma 0.5 \
    --model_save_path ./tmp/ \
    --prediction_max_len 64 \
    --dev_eval_path ./tmp/dev_pred.txt \
    --final_eval_path ./tmp/dev_eval.txt \
    --test_eval_path ./tmp/test_eval_%d.txt \
    --l2_lambda 1e-5 \
    --training_max_len 64 \
    --augment_percentage 0.2 \
    --augment_method by_rule \
    --augment_type insert \
    --augment_descending