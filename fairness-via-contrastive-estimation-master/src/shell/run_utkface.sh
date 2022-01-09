# ADV FORGETTING
bash src/shell/health/run_adv_forgetting_health.sh
python3 -m src.scripts.eval_embeddings -D -f result/health/adv_forgetting -r result/eval/health -m "adv_forgetting" --force -c config/eval_config.py
