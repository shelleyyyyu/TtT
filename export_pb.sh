gamma=0.5
dname="zh_full_merge_data"  # "SIGHAN15" "HybirdSet", "TtTSet"
dpath="./data/"$dname
bpath="./model/bert/"
cpath="./ckpt/"$dname"_"$gamma"/"

mkdir -p $cpath

python -u export_pb.py \
    --bert_path  $bpath/bert.ckpt\
    --bert_vocab $bpath/vocab_v2.txt \
    --train_data ./data/zh_full_merge_data/train_with_tag_info.txt \
    --dev_data ./data/zh_full_merge_data/test.txt\
    --test_data ./data/zh_full_merge_data/test.txt\
    --batch_size 1 \
    --lr 1e-5 \
    --dropout 0.1 \
    --number_epoch 10 \
    --gpu_id 0 \
    --print_every 1 \
    --save_every 1 \
    --fine_tune \
    --loss_type FC_FT_CRF_LABEL_FT\
    --gamma $gamma \
    --model_save_path $cpath \
    --prediction_max_len 48 \
    --dev_eval_path $cpath/dev_pred.txt \
    --final_eval_path $cpath/dev_eval.txt \
    --test_eval_path $cpath/test_eval_%d.txt \
    --l2_lambda 1e-5 \
    --training_max_len 48 \
    --restore_ckpt_path ./ckpt/ocr_epoch_27_dev_f1_0.855 \
    --onnx_path ./NLPCC/best_model