# ADV FORGETTING
bash src/shell/utkface/run_adv_forgetting_utkface.sh
python3 -m src.scripts.eval_embeddings -D -f result/utkface/adv_forgetting -r result/eval/utkface -m "adv_forgetting" --force -c config/eval_config.py
