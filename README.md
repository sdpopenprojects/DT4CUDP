# DT4CUDP

Replication package for *The Effect of Data Transformation Techniques on Clustering-based Unsupervised Software Defect Prediction: An Empirical Study* by Zhengxiang Chen, Zhiqiang Li, Hongyu Zhang, Jie Ren, and Feng Tian.

The scripts below rerun the **revised** protocol: clusters are fit on the transformed out-of-bag features, and defective clusters are labelled from the **original** metrics with `labelCluster_v4_raw` (no extra z-score inside the labeller). Bug labels are used only to score predictions.

## 1. Environment

Python 3.9. From the repository root:

```bash
pip install -r requirements.txt
```

`scikit-learn-extra` is required for K-medoids.

## 2. Data

The 22 paper projects are in `data/`.

| Group | Files | LOC column |
| --- | --- | --- |
| AEEEM | `EQ.arff`, `JDT.arff`, `ML.arff`, `PDE.arff`, `LC.arff` | `ck_oo_numberOfLinesOfCode` |
| PROMISE | `ant-1.7.csv`, `camel-1.4.csv`, `ivy-2.0.csv`, `jedit-4.0.csv`, `log4j-1.0.csv`, `poi-2.0.csv`, `tomcat.csv`, `velocity-1.6.csv`, `xalan-2.4.csv`, `xerces-1.3.csv` | `loc` |
| JIRA | `activemq-5.0.0.csv`, `derby-10.5.1.1.csv`, `groovy-1_6_BETA_1.csv`, `hbase-0.94.0.csv`, `hive-0.9.0.csv`, `jruby-1.1.csv`, `wicket-1.3.0-beta2.csv` | `CountLine` |

Check that every file is present:

```bash
python test/demo_NCIA.py --check
```

Expected output ends with `dataset check passed`.

## 3. Performance experiment

```bash
python test/demo_NCIA.py --smoke
python test/demo_NCIA.py
```

`--smoke` runs K-means on the original EQ data for 2 repetitions. The full command runs 23 clusterers × 6 transformations × 22 projects × 100 out-of-bag repetitions. The repetition index is the bootstrap seed (`utilities/CrossValidataion.py`).

Transformations: `O`, `log` (`log(x+1)`), `Z-score`, `Max-Min`, `yeo-johnson`, `rank-transformation`. Each scaler is fit on the current test subset.

Results are appended to:

```text
result/test_labelRaw/{algorithm}/{transformation}/{project}.csv
```

A finished file has 100 rows and no header:

`precision, recall, pf, F-measure, AUC, g-measure, g-mean, bal, MCC, Popt, Erecall, Eprecision, Efmeasure, PMI, IFA, time`

`AUC` is \((TPR+TNR)/2\). `Efmeasure` is F-measure@20% LOC and `IFA` is the initial false alarms, both from CBS+. Running the same command again skips any file that already has enough rows.

To split the 23 clusterers across machines, pass a comma-separated subset. These three groups do not overlap:

```bash
python test/demo_NCIA.py --clf Rock,AP,Gmeans,Cure,Optics,Dbscan,MeanShift,Syncsom
python test/demo_NCIA.py --clf Kmeans,MiniBatchKmeans,KmeansPlus,Birch,Agglomerative,Xmeans,Bsas,Mbsas,Ttsas
python test/demo_NCIA.py --clf Kmedoids,Fcm,EMA,GMM,SC,Somsc
```

Merge the three `result/test_labelRaw/` trees afterwards. Directories are separated by algorithm name.

## 4. Interpretation experiment

```bash
python test/demo_NCIA_ModeInter.py --smoke
python test/demo_NCIA_ModeInter.py
```

This runs K-means, K-medoids, G-means, AP, and spectral clustering. Each of 50 repetitions shuffles one feature at a time (5 shuffles), applies that same permutation to the original-metric column, **refits** the clusterer, and labels the new clusters with `labelCluster_v4_raw`. Each CSV row is the mean accuracy drop of one repetition, one value per feature:

```text
result/test_ModeInter_labelRaw/{algorithm}/{transformation}/{project}.csv
```

A subset uses the same `--clf` flag, for example `--clf AP,Gmeans`.

## 5. Layout

- `Cluster/` builds a clusterer from its name.
- `data/` holds the 22 datasets.
- `utilities/` holds the bootstrap split, labelling, measures, and permutation importance.
- `test/demo_NCIA.py` and `test/demo_NCIA_ModeInter.py` are the two entry points.
- `test/demo_DT.py` is an earlier driver and is not the revised experiment.

Scott-Knott ESD rankings and the top-k tables are computed from these CSV files after the runs finish. The two entry points do not write those rankings themselves.
