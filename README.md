# imbens-results
## Run procedure:
1. process_data.py
2. get_split_data.py
3. main.py
4. process_results.py

Datasets will be downloaded automatically and splitted using 5-fold cross-validation. Ten independent splits are performed. The number of base classifiers is determined using 3-fold cross-validation.

In total there are 15 imbalanced ensembles and 27 datasets included for comparisons in this study.
<img width="662" height="870" alt="image" src="https://github.com/user-attachments/assets/4b3c2f14-85cc-4cbd-987a-4290c2d2f1be" />

<img width="1253" height="501" alt="image" src="https://github.com/user-attachments/assets/55cbfc3c-b745-48be-9c8f-cf0726f4716f" />
Refer to the paper for more details.



## environment
Python=3.7.16

imbalanced-ensemble==0.1.7

or 

pip install -r requirements.txt


This should install all required packages.
