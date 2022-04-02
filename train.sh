# MODEL_NAME_OR_PATH=./dvdfinetunen/checkpoint-1500
MODEL_NAME_OR_PATH=xlm-roberta-base
#MODEL_NAME_OR_PATH=roberta-large
Seed=15
Domain=dvd

export CUDA_VISIBLE_DEVICES=0
nohup python train_sentiment.py \
  --do_train \
  --evaluate_during_training \
  --src_language en \
  --src_domain $Domain \
  --train_type  finetune \
  --labelword_ensemble true \
  --label0 "negative negativ négatif ネガティブ" \
  --label1 "positive positiv positif ポジティブ" \
  --head_type  mlm \
  --data_type  few \
  --n_shot 32 \
  --n_class 2 \
  --model_type xlmr \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --max_len 512 \
  --per_gpu_train_batch_size  4 \
  --per_gpu_eval_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --logging_steps 1 \
  --seed $Seed \
  --eval_all_checkpoints \
  --learning_rate 5e-5 \
  --num_warmup_steps 200 \
  --num_train_epochs 50 \
  --save_steps 0 > result/32shot_cls_mixlabelword_test_${Domain}_${Seed}.log &

# negativ négatif ネガティブ   отрицательный（俄语） negativní（捷克语） 부정적인 négative（法语阴性）
# positiv positif ポジティブ   положительный   pozitivní    긍정적인 
