import sys
import time
import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

train_csv_path = os.path.join(current_dir, 'train.csv')
test_csv_path = os.path.join(current_dir, 'test_local.csv')

if __name__ == "__main__":
    from solution import Solution

    forbidden = ['xgboost']
    imported = list(sys.modules.keys())
    violations = [module for module in imported
                  if any(module.lower() == pkg or module.lower().startswith(pkg + '.')
                         for pkg in forbidden)]

    if violations:
        print(f"Error: Forbidden libraries detected: {violations}")
        sys.exit(1)

    solution = Solution()
    train_df = pd.read_csv(train_csv_path)

    print(f"Training")
    y_train = train_df['label'].values
    X_train = train_df.drop('label', axis=1)

    start_time = time.time()
    solution.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f}s")

    test_df = pd.read_csv(test_csv_path)
    y_true = test_df['label'].values
    test_features = test_df.drop('label', axis=1)

    samples = [(idx, row.to_dict()) for idx, row in test_features.iterrows()]
    predictions = [None] * len(samples)
    probabilities = [None] * len(samples)


    def process_sample(sample_info):
        idx, sample = sample_info
        result = solution.forward(sample)
        return idx, result['prediction'], result['probability']


    print(f"Testing")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(process_sample, samples)
        for idx, pred, prob in results:
            predictions[idx] = pred
            probabilities[idx] = prob
    testing_time = time.time() - start_time

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    roc_auc = roc_auc_score(y_true, probabilities)
    latency = np.sqrt(training_time * testing_time)

    print(f"\n{'=' * 50}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Testing Time:  {testing_time:.2f}s")
    print(f"Latency:   {latency:.2f}s")
    print(f"ROC-AUC:       {roc_auc:.6f}")
    print(f"{'=' * 50}\n")