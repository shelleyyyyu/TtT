gamma=0.5
dpath="./data/NLPCC"
bpath="./model/bert/"
cpath="./ckpt/NLPCC_"$gamma"/"

mkdir -p $cpath

python -u main.py \
    --bert_path  $bpath/bert.ckpt\
    --bert_vocab $bpath/vocab_v2.txt \
    --train_data ./data/NLPCC/nlpcc/train.txt \
    --dev_data ./data/NLPCC/nlpcc/test.txt\
    --test_data ./data/NLPCC/nlpcc/test.txt\
    --batch_size 1 \
    --lr 1e-5 \
    --dropout 0.1 \
    --number_epoch 30 \
    --gpu_id 0 \
    --print_every 1 \
    --save_every 1 \
    --fine_tune \
    --loss_type FC_FT_CRF_LABEL_FT\
    --gamma $gamma \
    --model_save_path $cpath \
    --prediction_max_len 64 \
    --dev_eval_path $cpath/dev_pred.txt \
    --final_eval_path $cpath/dev_eval.txt \
    --test_eval_path $cpath/test_eval_%d.txt \
    --l2_lambda 1e-5 \
    --training_max_len 64