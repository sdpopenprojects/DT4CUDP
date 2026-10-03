# General Introduction

This repository provides the code and datasets used in the article: *"The Effect of Data Transformation Techniques on Clustering-based Unsupervised Software Defect Prediction: An Empirical Study"* by Zhengxiang Chen, Zhiqiang Li, Hongyu Zhang, Jie Ren, and Feng Tian, submitted to the *Automated Software Engineering* Journal.

### Environment Preparation

- The study requires Python 3.9 with specific library versions.

```
numpy>=1.20.0

pandas>=1.3.0

scipy>=1.7.0

scikit-learn>=1.0.0

pyclustering>=0.10.0
```

### Repository Structure

- `Cluster/` : Create and return the corresponding clustering algorithm model instance based on the input model name.

- `data/` : Contains all experimental datasets ('.arff' and '.csv' files).

- `utilities/` : Utility modules for cross-validation, label assignment, performance measurement, ranking, model interpretation, etc.

- `test/` : Contains the main programs that drive the experiments.

### Run the Main Experiments

Navigate to the `test/` directory and execute the main programs. The repository provides two entry points:

\# Run experiments for defect prediction performance

`python demo_NCIA.py`

\# Run experiments for model interpretation

`python demo_NCIA_ModeInter.py`

Each script will:
1)	Load the relevant datasets from the `data/` directory.
2)	Apply the corresponding data transformation technique.
3)	Instantiate the clustering model via `Cluster/GetCluster.py`.
4)	Perform clustering and label clusters using the `utilities/` scripts.
5)	Compute evaluation measures for model performance using the `utilities/` scripts for `demo_NCIA.py`.
6)	Compute feature importance scores for model interpretation using the `utilities/` scripts for `demo_NCIA_ModeInter.py`.
7)	Generate ranked lists of performance and feature importance based on the double NPSKESD test.

