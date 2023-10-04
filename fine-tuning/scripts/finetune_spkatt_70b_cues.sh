#CUDA_VISIBLE_DEVICES=1,2,3 \
python ../../../qlora.py \
    --model_name_or_path path_to_llama-2_models/70b \
    --output_dir ./output/spkatt-70b-cues \
    --data_seed 42 \
    --save_steps 500 \
    --evaluation_strategy no \
    --dataloader_num_workers 4 \
    --lora_modules all \
    --bf16 \
    --dataset="parsed_data_cues.jsonl" \
    --dataset_format="input-output" \
    --source_max_len 256 \
    --target_max_len 64 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_steps 2500 \
    --learning_rate 0.0001 \
    --lora_dropout 0.05 \
    --seed 0


# Die "source" ist ein Teil des Prompts am Anfang, der aus dem Training ausgelassen wird. Dies können z.B. System Prompts sein.
# target_max_len may need to be higher
