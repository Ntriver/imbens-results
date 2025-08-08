import numpy as np
import os
import pandas as pd

ensemble_list = [
                'SelfPacedEnsembleClassifier',
                 'BalanceCascadeClassifier',
                 'BalancedRandomForestClassifier',
                 'EasyEnsembleClassifier',
                 'RUSBoostClassifier',
                 'UnderBaggingClassifier',
                 'OverBoostClassifier',
                 'SMOTEBoostClassifier',
                 # 'KmeansSMOTEBoostClassifier',# not working well
                 'OverBaggingClassifier',
                 'SMOTEBaggingClassifier',
                 'AdaCostClassifier',
                 'AdaUBoostClassifier',
                 'AsymBoostClassifier',
                 'CompatibleAdaBoostClassifier',
                 'CompatibleBaggingClassifier'
                 ]

result_root = './results/'
all_file_names = os.listdir(result_root)
all_file_names.remove('protein_homo')
all_file_names.remove('webpage')
all_file_names.remove('isolet')

metric_names = "acc, ba, f1, mc, auroc, auprc, p_pos, r_pos, p_neg, r_neg".split(sep=', ')

metrics_data = {metric: np.zeros((27, 15)) for metric in metric_names}

for file_idx, file_name in enumerate(all_file_names):#### -12==protein_homo， -4webpage, 7 isolet
    for ensemble_idx, ensemble_name in enumerate(ensemble_list):
        print(file_name, ensemble_name)
        npyfile = np.load(os.path.join(result_root,file_name,ensemble_name+'.npy'))
        res = npyfile.reshape(10,5,10).mean(axis=1).mean(axis=0) # (10,)

        for metric_idx in range(10):
            metrics_data[metric_names[metric_idx]][file_idx, ensemble_idx] = res[metric_idx]

# 为每个指标创建DataFrame并保存
output_dir = 'performance_tables'
os.makedirs(output_dir, exist_ok=True)

for metric_name, data in metrics_data.items():
    df = pd.DataFrame(
        data,
        index=all_file_names,
        columns=ensemble_list
    )

    # 保存为CSV文件
    output_file = os.path.join(output_dir, f"{metric_name}_performance.csv")
    df.to_csv(output_file)
    print(f"Saved {output_file}")

print("所有性能表格已生成！")

