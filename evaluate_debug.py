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

    # Use a smaller subset for quick testing of latency improvement
    # But for final score I need full set.
    # Let's just run on full set but print progress.
    
    # samples = [(idx, row.to_dict()) for idx, row in test_features.iterrows()]
    # Faster way to create samples list
    print("Preparing samples...")
    samples = []
    dicts = test_features.to_dict('records')
    for idx, d in enumerate(dicts):
        samples.append((idx, d))
    print("Samples prepared.")
    predictions = [None] * len(samples)
    probabilities = [None] * len(samples)

    def process_sample(sample_info):
        idx, sample = sample_info
        result = solution.forward(sample)
        return idx, result['prediction'], result['probability']

    print(f"Testing {len(samples)} samples...")
    start_time = time.time()
    
    # Use map directly without executor to debug speed if needed, but executor is required by task structure simulation
    # Actually evaluate_local.py uses ThreadPoolExecutor.
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(process_sample, samples)
        for i, (idx, pred, prob) in enumerate(results):
            predictions[idx] = pred
            probabilities[idx] = prob
            if i % 10000 == 0:
                print(f"Processed {i} samples...", end='\r')
                
    testing_time = time.time() - start_time
    print(f"\nTesting completed in {testing_time:.2f}s")

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
