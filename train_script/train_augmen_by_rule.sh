augment_percentage="0.1"
augment_type="contributed"
augment_method="by_rule"
suffix="augment_"$augment_percentage"_"$augment_type"_"$augment_method"_"
bpath="./model/bert/"
cpath="/data/yumenghsuan/cgec/ckpt/NLPCC_"$suffix"/"

mkdir -p $cpath

nohup python -u main.py \
    --bert_path  $bpath/bert.ckpt\
    --bert_vocab $bpath/vocab.txt \
    --train_data /data/yumenghsuan/cgec_data/nlpcc/train.txt \
    --dev_data /data/yumenghsuan/cgec_data/nlpcc/test.txt\
    --test_data /data/yumenghsuan/cgec_data/nlpcc/test.txt\
    --batch_size 80 \
    --lr 1e-5 \
    --dropout 0.1 \
    --number_epoch 30 \
    --gpu_id 1 \
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
    --augment_percentage $augment_percentage \
    --augment_type $augment_type \
    --augment_method $augment_method \
    --augment_descending > ./logs/train.nlpcc.$suffix.log 2>&1&
