### evaluation
python main.py --save_dir ./eval/CUFED/TTSR \
               --reset True \
               --log_file_name eval.log \
               --eval True \
               --eval_save_results True \
               --num_workers 4 \
               --dataset CUFED \
               --dataset_dir  /home/ivclab2/TTSR_FFT/dataset/CUFED \
               --model_path /model_00050.pt