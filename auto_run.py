# -*- coding: utf-8 -*-
import logging
import subprocess
import time
from itertools import product

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # 配置日志记录器

    l = ['tencent','wangyi','zongxiang']
    batch_sizes = {32}
    learning_rates = {'2e-5'}
    shots = {20}
    seeds = {123}
    template_id = {0}
    verbalizer = {'cpt'}
    for n, t, j, i, k, m, v in product(l,template_id, seeds, batch_sizes, learning_rates, shots, verbalizer):
        cmd = (
            f"python fewshot.py --result_file ./output_fewshot.txt "
            f"--dataset {n} --template_id {t} --seed {j} "
            f"--batch_size {i} --shot {m} --learning_rate {k} --verbalizer {v}"
        )

        logging.info(f"Executing command: {cmd}")
        print(cmd)
        try:
            subprocess.run(cmd, shell=True, check=True)
            logging.info(f"Command executed successfully: {cmd}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed: {cmd}. Error: {e}")

        time.sleep(2)