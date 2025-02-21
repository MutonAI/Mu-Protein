import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import argparse
import os
import glob
from datetime import datetime


def cumulative_max_per_round(sequences):
    num_rounds = sequences['round'].max() + 1
    max_per_round = [sequences['true_score'][sequences['round'] == r].max()
                     for r in range(num_rounds)]

    return np.maximum.accumulate(max_per_round)


def get_latest_csv(directory, name_pattern):
    """获取指定目录下最新的CSV文件
    优先使用带时间戳的文件，如果没有则使用最后修改时间"""
    # 获取目录下所有csv文件
    pattern = os.path.join(directory, "*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"在 {directory} 目录下没有找到CSV文件")
    
    # 尝试找出带有时间戳格式的文件
    timestamped_files = []
    other_files = []
    
    for file in csv_files:
        try:
            # 尝试解析文件名中的时间戳
            filename = os.path.basename(file)
            timestamp_str = filename.split('*')[-1].split('.')[0]
            datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            # print(f"file: {file}, name_pattern: {name_pattern}")
            if name_pattern in file:
                timestamped_files.append(file)
        except (IndexError, ValueError):
            # 如果解析失败，将文件添加到其他文件列表
            # print(f"file: {file}, name_pattern: {name_pattern}")
            if name_pattern in file:
                other_files.append(file)
    
    if timestamped_files:
        return timestamped_files
    else:
        # 如果没有带时间戳的文件，使用最后修改时间最新的文件
        return other_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn', action='store_true')
    args = parser.parse_args()
    # for landscape in ['muformer']:
    if args.cnn:
        landscapes = ['aav']
    else:
        landscapes = ['muformer']
    for landscape in landscapes:
        for sequences_batch_size, model_queries_per_batch in [(100, 5000), (1000, 5000)]:
            plt.figure(dpi=300)
            plt.title(f'Performance with respect to {sequences_batch_size} and {model_queries_per_batch} on {landscape}')
            if args.cnn:
                # method_list = ['adalead', 'cbas', 'cmaes', 'dynappo', 'BO', 'dirichlet_ppo_cnn', 'dirichlet_ppo_update_starting_sequence_cnn', 'test_cnn']
                # dirichlet_ppo_list = ['dirichlet_ppo_cnn', 'dirichlet_ppo_update_starting_sequence_cnn', 'test_cnn']
                method_list = ['dirichlet_ppo_cnn']
                # method_list = ['adalead', 'cbas', 'cmaes', 'dynappo', 'BO', 'dirichlet_ppo_cnn']
                dirichlet_ppo_list = ['dirichlet_ppo_cnn']
            else:
                method_list = ['adalead_gt', 'cmaes_gt', 'dynappo_gt', 'BO_gt', 'dirichlet_ppo']
                # method_list = ['adalead_gt', 'cbas_gt', 'cmaes_gt', 'dynappo_gt', 'BO_gt', 'dirichlet_ppo']
                dirichlet_ppo_list = ['dirichlet_ppo']
            for method in method_list:
            # for method in ['adalead_gt', 'cbas_gt', 'cmaes_gt', 'dynappo_gt', 'BO_gt', 'dirichlet_ppo', 'dirichlet_ppo_update_starting_sequence', 'test']:
                if method in dirichlet_ppo_list:
                # if method in ['dirichlet_ppo', 'dirichlet_ppo_update_starting_sequence', 'test']:
                    if args.cnn:
                        file_names = get_latest_csv(f'/home/v-zhaozhiyu/code/FLEXS/efficiency/{method}/{landscape}', str(sequences_batch_size)+'_'+str(model_queries_per_batch))
                        # file_name = f'/home/v-zhaozhiyu/code/FLEXS/efficiency/{method}/{landscape}/{sequences_batch_size}_{model_queries_per_batch}.csv'
                    else:
                        # if method == 'dirichlet_ppo' and landscape == 'rna':
                        #     file_name = f'/home/v-zhaozhiyu/code/FLEXS/efficiency/dirichlet_ppo/rna/100_5000_20250118_094302.csv'
                        file_names = get_latest_csv(f'/home/v-zhaozhiyu/code/FLEXS/efficiency/{method}/{landscape}', str(sequences_batch_size)+'_'+str(model_queries_per_batch))
                        # file_name = f'/home/v-zhaozhiyu/code/FLEXS/efficiency/{method}/{landscape}/{sequences_batch_size}_{model_queries_per_batch}_new.csv'
                else:
                    file_names = get_latest_csv(f'/home/v-zhaozhiyu/code/FLEXS/efficiency/{method}/{landscape}', str(sequences_batch_size)+'_'+str(model_queries_per_batch))
                for file_name in file_names:
                    with open(file_name) as f:
                        metadata = json.loads(next(f))
                        data = pd.read_csv(f)

                        rounds = data['round'].unique()
                        max_per_round = cumulative_max_per_round(data)
                        # plt.plot(rounds, max_per_round, '-o', label=f'{method}')
                    try:
                        plt.plot(rounds, max_per_round, '-o', label=f'{method}')
                    except Exception as e:
                        print(file_name)
                        print(f"rounds: {rounds}, max_per_round: {max_per_round}")
                        print(e)                
                        pass
                    plt.ylabel("Cumulative max")
                    plt.xlabel("Round")

            plt.legend()
            plt.savefig(f"examples/figs/dirichlet_ppo_cnn_{landscape}_{sequences_batch_size}_{model_queries_per_batch}.png")
            # plt.savefig(f'examples/figs/efficiency_{landscape}_{sequences_batch_size}_{model_queries_per_batch}.png')
            # if args.cnn:
            #     plt.savefig(f'examples/figs/cnn/efficiency_{landscape}_{sequences_batch_size}_{model_queries_per_batch}_cnn.png')
            # else:
            #     plt.savefig(f'examples/figs/gt/efficiency_{landscape}_{sequences_batch_size}_{model_queries_per_batch}_gt_updated.png')