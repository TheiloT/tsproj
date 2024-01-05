import numpy as np
import os
import pandas as pd


def fetch_metric(path, metric="accuracy", mode="test"):
    """
    Fetch accuracy stored in experiment folder indicated by path, for train or test mode.
    """
    df = pd.read_csv(os.path.join(path, f"perf-{mode}.csv"))
    return (df[metric].values, df.index.values)


def average_multi_run_perfs(dataset, experiment, metric="accuracy", interpolations=np.array([0, 10]), seeds=np.arange(1, 6), mode="test"):
    """
    Average metrics over multiple runs of the same experiment.
    """
    exp_path = os.path.join("outputs", f"{dataset}_experiments", f"experiment_{experiment}", "multi")
    # Fetch number of units
    units = fetch_metric(os.path.join(exp_path, f"interp_{interpolations[0]}", f"seed_{seeds[0]}"), metric=metric, mode=mode)[1]
    
    # Create an empty dataframe to store the accuracy values
    metric_df = pd.DataFrame(0, index=units, columns=interpolations)
    metric_df.name = metric
    metric_df.columns.name = "interpolation"
    metric_df.index.name = "unit"
    
    # Iterate over each interpolation value
    for interpolation in interpolations:
        # Average over seed values
        for seed in seeds:
            accuracy_by_unit = fetch_metric(os.path.join(exp_path, f"interp_{interpolation}", f"seed_{seed}"), metric=metric, mode=mode)[0]
            for i, unit in enumerate(units):
                metric_df.loc[unit, interpolation] += accuracy_by_unit[i]
        metric_df.loc[:, interpolation] /= len(seeds)
    
    print(f"Averaged {metric}:")
    return metric_df