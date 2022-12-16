"""
**Distribution A**
Approved for Public Release, Distribution Unlimited
"""

import os
import gzip
import json
import time
import itertools
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.utils._testing import ignore_warnings
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty, AtomicOrbitals, BandCenter

# directory for saving figures
FIG_BASEPATH = os.path.join("data", "fig")

# create directory for storing figures if it doesn't already exist
if not os.path.isdir(FIG_BASEPATH):
    FIG_BASEPATH = "fig"
    os.makedirs(FIG_BASEPATH)

FONTSIZE = 10
LINEWIDTH = 1
TICKWIDTH = 1
plt.rcParams.update(
    {
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "axes.linewidth": LINEWIDTH,
        "xtick.minor.width": TICKWIDTH,
        "xtick.major.width": TICKWIDTH,
        "ytick.minor.width": TICKWIDTH,
        "ytick.major.width": TICKWIDTH,
        "font.family": "Arial",
        "figure.facecolor": "w",
        "figure.dpi": 600,
    }
)


def write_jsonzip(data: dict, filepath: str):
    """Write json data to a zipped json file"""
    with gzip.open(filepath, "w") as fout:
        fout.write(json.dumps(data).encode("utf-8"))


def read_jsonzip(filepath: str) -> dict:
    with gzip.open(filepath, "r") as fin:
        data = json.loads(fin.read().decode("utf-8"))
    return data


def array_is_constant(arr: list) -> np.array:
    """Check if a 1D array is constant"""
    return np.allclose(arr, np.repeat(arr[0], len(arr)))


def norm_array(a: list) -> np.array:
    """Normalize a 1D array from 0 to 1"""
    a = np.asarray(a)
    if np.ptp(a) == 0:
        return a
    else:
        return (a - a.min()) / (np.ptp(a))


def standardize_df(
    df: pd.DataFrame, ignore: list = [], scaler=RobustScaler()
) -> pd.DataFrame:
    """Standardize data in a dataframe"""
    dfs = df[get_num_cols(df, ignore=ignore)]
    dfs = pd.DataFrame(
        index=dfs.index,
        data=scaler.fit_transform(dfs.values),
        columns=dfs.columns,
    )
    return dfs


def get_tsne(
    df: pd.DataFrame, ignore: list = [], perplexity: int = 30, n_components: int = 2
) -> pd.DataFrame:
    """
    Get t-sne embedding of dataframe. Return encoded dataframe.
    Ignore column names in the *ignore* list argument when
    performing transformation.
    """
    # standardize dataframe
    dfs = standardize_df(df[get_num_cols(df, ignore=ignore)])
    # get t-sne embedding
    ts = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=0,
        init="pca",
        learning_rate="auto",
    ).fit_transform(dfs.values)
    # put results back into another dataframe
    ts = pd.DataFrame(columns=["x", "y"], data=ts, index=dfs.index)
    # normalize values
    ts["x"] = norm_array(ts["x"])
    ts["y"] = norm_array(ts["y"])
    return ts


def get_num_cols(df: pd.DataFrame, ignore: list = []) -> list:
    """From a dataframe, get a list of all numeric columns,
    excluding column names in the ignore argument"""
    return [
        c
        for c in df
        if all(
            [
                pd.api.types.is_numeric_dtype(df[c]),
                c not in ignore,
            ]
        )
    ]


def unique_enough(arr: list, threshold: int = 5) -> bool:
    """
    Determine if an array contains enough unique values.
    For example, if threshold=10, then the array
    is unique enough if it contains at least 10 unique values.
    """
    return len(np.unique(arr)) >= threshold


def get_correlated_cols(
    df: pd.DataFrame, ignore: list = [], threshold: float = 0.98
) -> list:
    """
    Get the columns of a dataframe which have high
    Pearson correlation (r^2 > threshold).
    """
    cols = [
        c
        for c in df
        if all(
            [
                pd.api.types.is_numeric_dtype(df[c]),
                c not in ignore,
            ]
        )
    ]
    corr = []
    pairs = list(itertools.combinations(cols, 2))
    for p in pairs:
        x, y = df[p[0]].values, df[p[1]].values
        r = np.square(stats.pearsonr(x, y)[0])
        if r > threshold:
            print("CORRELATED:", p, round(r, 2))
            corr.append(p)
    return corr


def get_identical_cols(df: pd.DataFrame) -> dict:
    """
    Get the columns of a dataframe which have identical values.
    """
    xx = {}
    for c in df:
        vals1 = list(df[c])
        for c2 in df:
            vals2 = list(df[c2])
            # if the columns contain identical values
            if c != c2 and vals1 == vals2:
                # if these values have not been seen before
                if vals1 not in [list(df[y]) for y in xx]:
                    xx[c] = [c2]
                # if these values have been seen before
                else:
                    y = [yy for yy in xx if list(df[yy]) == vals1][0]
                    if c2 != y and c2 not in xx[y]:
                        xx[y].append(c2)
    return xx


def find_outlier_rows(
    df: pd.DataFrame, threshold: int = 3, verbose: bool = False
) -> list:
    """
    Identify the outlier rows in a dataframe using the
    median value of each column and the interdecile range of
    each column. Rows which contain at least one value
    which is further away from the median by a threshold
    amount of interdecile ranges are considered outliers.
    """
    # decile ranges of each column
    col_idr = stats.iqr(df.values, axis=0, rng=(10, 90))
    # median values of each column
    col_median = np.median(df.values, axis=0)
    outlier_rows = []
    # loop over each row and check if its values violate the threshold
    for row_i, row in enumerate(df.values):
        # check if any row values are too high or too low
        vals_too_high = row > col_median + (threshold * col_idr)
        vals_too_low = row < col_median - (threshold * col_idr)
        if any(vals_too_high) or any(vals_too_low):
            outlier_rows.append(df.index[row_i])
    if verbose:
        print(f"found {len(outlier_rows)} outlier rows")
    return outlier_rows


def create_datasets():
    """
    Aggregate datasets from file and plot histograms of their
    target variables.
    """
    # read the configuration file
    config = pd.read_csv(os.path.join("data", "dataset_config.csv"), index_col="name")
    config = config.fillna("")

    # configure plot
    fig, ax = plt.subplots(nrows=3, ncols=3)
    ax = np.ravel(ax)
    ii = 0
    ALPHA = 0.6
    TYPE_COLORS = {
        "materials, Magpie": "green",
        "materials, non-Magpie": "dodgerblue",
        "non-materials": "tomato",
    }

    ds = {}

    # loop over each row of the configuration file
    for dsn, row in config.iterrows():

        # read dataset
        print(dsn)
        df = pd.read_csv(os.path.join("data", "raw", dsn + ".csv"))
        t = row["target"]

        # drop extreme outlier rows
        outlier_rows = find_outlier_rows(df[get_num_cols(df)], threshold=3)
        df = df.drop(outlier_rows)
        df = df.drop_duplicates()

        # use magpie to featurize some of the datasets
        if row["type"] == "materials, Magpie":
            df = featurize(df)

        # remove non-numeric columns
        df = df[[c for c in df if pd.api.types.is_numeric_dtype(df[c])]]
        # remove columns with not many unique values
        df = df[[c for c in df if unique_enough(df[c].values)]]
        # remove constant columns
        df = df.loc[:, (df != df.iloc[0]).any()]
        # change all infinite values to nan
        df = df.replace([np.inf, -np.inf], np.nan)
        # remove rows containing a nan
        df = df.dropna()
        # get columns which contain identical values and remove them
        remove_cols = [v0 for _, v in get_identical_cols(df).items() for v0 in v]
        df = df[[c for c in df if c not in remove_cols]]

        # save data to file
        print(dsn, df.values.shape, f"---> target: {t}", "\n")
        ds[dsn] = {"df": df.to_json(default_handler=str)}
        for k in list(config):
            ds[dsn][k] = row[k]

        # plot histogram of target variable
        std = df[t].std()
        median = df[t].median()
        ax[ii].axvline(x=median, lw=0.75, linestyle="dashed", c="k")
        ax[ii].axvline(x=median - std, lw=0.5, linestyle="dotted", c="dimgray")
        ax[ii].axvline(x=median + std, lw=0.5, linestyle="dotted", c="dimgray")
        ax[ii].hist(
            df[t], bins=10, color=TYPE_COLORS[row["type"]], alpha=ALPHA, linewidth=0
        )
        xlabel = f"{t} ({row['units']})" if row["units"] else t
        ax[ii].set_xlabel(xlabel, fontsize=FONTSIZE)
        if ii in [0, 3, 6]:
            ax[ii].set_ylabel("Counts", fontsize=FONTSIZE)
        ax[ii].text(
            0.98,
            0.96,
            f"n={len(df)}",
            fontsize=FONTSIZE - 2,
            ha="right",
            va="top",
            transform=ax[ii].transAxes,
        )

        ii += 1

    ax[1].legend(
        ncol=3,
        bbox_to_anchor=(0.5, 1.05),
        loc="lower center",
        handles=[
            mpatches.Patch(
                color=TYPE_COLORS[k],
                alpha=ALPHA,
                label=k,
            )
            for k in TYPE_COLORS
        ],
    )

    plt.subplots_adjust(wspace=-1.2, hspace=0.4)
    plt.tight_layout()
    fig.savefig(
        os.path.join(FIG_BASEPATH, "DatasetHistograms.png"), bbox_inches="tight"
    )
    plt.show()

    # save datsets to file
    write_jsonzip(ds, os.path.join("data", "datasets.json.gz"))


def format_labels(labs0):
    """Format an iterable of variable labels for plotting"""
    if isinstance(labs0, list):
        labs = []
        for l0 in labs0:
            l = l0.replace("MagpieData ", "")
            l = (
                l.replace("minimum", "min")
                .replace("maximum", "max")
                .replace("Number", "Num")
                .replace("Electronegativity", "Electroneg")
            )
            if "**" in l:
                # remove unnecessary decimals in exponents
                base, power = l.split("**")
                if float(power).is_integer():
                    power = int(float(power))
                    l = f"{base}**{power}"
                l = l.replace("**", "$^{") + "}$"
            labs.append(l)
        return labs
    else:
        l = labs0.replace("MagpieData ", "")
        l = (
            l.replace("minimum", "min")
            .replace("maximum", "max")
            .replace("Number", "Num")
            .replace("Electronegativity", "Electroneg")
        )
        if "**" in l:
            # remove unnecessary decimals in exponents
            base, power = l.split("**")
            if float(power).is_integer():
                power = int(float(power))
                l = f"{base}**{power}"
            l = l.replace("**", "$^{") + "}$"
        return l


def refactor_feat(fl):
    """Refactor a feature list so each variable only appears once"""
    if type(fl) == str:
        return fl
    base_vars = set([v.split("**")[0] for v in fl])
    fl2 = {}
    # loop over all base variables
    for bv in base_vars:
        # loop over all factors in the feature
        for f in fl:
            # if the base variable is in the factor
            if bv in f:
                # if its raised to a power
                if len(f.split("**")) > 1:
                    power = float(f.split("**")[1])
                    if bv not in fl2:
                        fl2[bv] = power
                    else:
                        fl2[bv] = fl2[bv] + power
                # if variable is not raised to a power
                else:
                    if bv not in fl2:
                        fl2[bv] = 1
                    else:
                        fl2[bv] = fl2[bv] + 1
        if fl2[bv] == 0:
            fl2.pop(bv, None)
        else:
            fl2[bv] = round(fl2[bv], 3)
    # return in list form and sort
    fl2 = [f"{k}**{v}" if v != 1 else k for k, v in fl2.items()]
    fl2.sort()
    return fl2


def print_dict_structure(d: dict, kk: str = None, indent: int = 2):
    """Print the structure of a large nested dict"""
    if isinstance(d, dict):
        for k in list(d):
            print(" " * indent, k, ":", type(d[k]).__name__)
            if isinstance(d[k], dict):
                print_dict_structure(d[k], kk=k, indent=indent + 2)


def featurize(
    df: pd.DataFrame,
    formula_col: str = "formula",
    pbar: bool = False,
    n_jobs: int = None,
    n_chunksize: int = None,
) -> pd.DataFrame:
    """
    Featurization of chemical formulas for machine learning.
    Input a Pandas Dataframe with a column called formula_col,
    which contains chemical formulas (e.g. ['Mg', 'TiO2']).
    Other columns may contain additional descriptors.
    The chemical formulas are featurized according to methods
    from the matminer package. Returns the dataframe with chemical
    formulas and features, and a list of references
    to papers which describe the featurization methods used.

    To prevent issues with multithreading, set n_jobs=1.

    To ignore certain features, comment them out in the
    'composition_features' or 'composition_ox_features' lists.

    Use the kwargs to return the list of references used,
    remove dataframe columns which are constant or
    remove dataframe columns which contain nans.

    ================= Useful links =======================
    Matminer summary table of features:
    https://hackingmaterials.lbl.gov/matminer/featurizer_summary
    Matminer Github repo:
    https://github.com/hackingmaterials/matminer
    Matminer notebook examples:
    https://github.com/hackingmaterials/matminer_examples
    """
    starttime = time.time()
    if formula_col not in list(df):
        raise KeyError("Data does not contain {} column.".format(formula_col))

    print("Featurizing dataset...")
    # create composition column from the chemical formula
    stc = StrToComposition()
    if n_jobs:
        stc.set_n_jobs(n_jobs)

    feat = stc.featurize_dataframe(df, formula_col, ignore_errors=True, pbar=pbar)

    # add element property featurizer
    element_property = ElementProperty.from_preset(preset_name="magpie")

    # loop over each feature and add it to the dataframe
    for f in [
        element_property,
        BandCenter(),
        AtomicOrbitals(),
    ]:
        # n_jobs = 1 is required to prevent multithread hanging on large molecules
        if n_jobs:
            f.set_n_jobs(n_jobs)
        if n_chunksize:
            f.set_chunksize(n_chunksize)

        # implement feature
        feat = f.featurize_dataframe(feat, "composition", pbar=pbar, ignore_errors=True)

    # set dataframe index as chemical formula
    feat = feat.set_index(feat[formula_col])
    num_new_features = len(list(feat))
    print(f"Featurization time: {round((time.time() - starttime) / 60, 2)} min")
    return feat


def get_r2(x: list, y: list) -> float:
    """Get the Peason r^2 correlation coefficient"""
    return np.square(stats.pearsonr(x, y)[0])


def plot_feature_correlations_w_targets():
    """Plot Pearson and Spearman correlations between
    all input features and the target feature"""
    fig, ax = plt.subplots(nrows=3, ncols=3)
    ax = np.ravel(ax)
    ii = 0
    CV_COLORS = {
        "Original": "black",
        "LOCO": "tomato",
        "Random": "dodgerblue",
    }
    CV_COLORS2 = {
        "Original": "black",
        "LOCO": "red",
        "Random": "blue",
    }
    ALPHA = 0.2

    ds = read_jsonzip(os.path.join("data", "datasets.json.gz"))
    fs = read_jsonzip(os.path.join("data", "features.json.gz"))

    for dsn in ds:
        print(dsn)
        t = ds[dsn]["target"]
        df = pd.DataFrame(json.loads(ds[dsn]["df"]))

        for split_type in ["Original", "LOCO", "Random"]:
            r2s, sr2s = [], []

            if split_type == "Original":
                for c in get_num_cols(df, ignore=[t]):
                    x, y = df[c].values, df[t].values
                    r2s.append(np.square(stats.pearsonr(x, y)[0]))
                    sr2s.append(np.square(stats.spearmanr(x, y)[0]))
            else:
                # get pearson and spearman r^2 values across all new features
                new_feat_df = pd.concat(
                    [pd.DataFrame(json.loads(i)) for i in fs[dsn][split_type]]
                )
                for i, row in new_feat_df.iterrows():
                    x, y = row["val"], df[t].values
                    r2s.append(np.square(stats.pearsonr(x, y)[0]))
                    sr2s.append(np.square(stats.spearmanr(x, y)[0]))

            # plot all features
            ax[ii].scatter(
                sr2s,
                r2s,
                lw=0.25,
                s=6,
                edgecolors="w",
                alpha=ALPHA,
                c=CV_COLORS[split_type],
            )

            # plot median feature location
            ax[ii].scatter(
                np.median(sr2s),
                np.median(r2s),
                s=100,
                alpha=1,
                c=CV_COLORS2[split_type],
                marker="+",
                lw=1,
                zorder=99,
            )

        ax[ii].tick_params(labelsize=FONTSIZE)
        # ax[ii].set_xticks([0, 0.5, 1], [0, 0.5, 1])
        # ax[ii].set_yticks([0, 0.5, 1], [0, 0.5, 1])
        ax[ii].set_xlim([0, 1])
        ax[ii].set_ylim([0, 1])

        if ii in [0, 3, 6]:
            ax[ii].set_ylabel("r$^2$", fontsize=FONTSIZE)

        else:
            ax[ii].set_ylabel("")
            ax[ii].set_yticklabels([])
        if ii in [6, 7, 8]:
            ax[ii].set_xlabel("ρ$^2$", fontsize=FONTSIZE)
        else:
            ax[ii].set_xlabel("")
            ax[ii].set_xticklabels([])

        ax[ii].text(
            0.03,
            0.96,
            t,
            fontsize=FONTSIZE,
            ha="left",
            va="top",
            transform=ax[ii].transAxes,
        )

        ax[ii].axvline(
            x=0.5, color="gray", lw=1, linestyle="dotted", alpha=0.3, zorder=0
        )
        ax[ii].axhline(
            y=0.5, color="gray", lw=1, linestyle="dotted", alpha=0.3, zorder=0
        )
        ii += 1

    ax[1].legend(
        ncol=3,
        bbox_to_anchor=(0.5, 1.05),
        loc="lower center",
        # fontsize=16,
        handles=[
            mpatches.Patch(
                color=CV_COLORS[k],
                alpha=1,
                label=k,
            )
            for k in CV_COLORS
        ],
    )
    plt.subplots_adjust(wspace=-1.2, hspace=0.4)
    plt.tight_layout()
    plt.gcf().savefig(os.path.join(FIG_BASEPATH, "FeatureCorrelations.png"))
    plt.show()


def get_random_splits(
    a: list, n_splits: int = 10, split_frac: tuple = (0.8, 0.1, 0.1)
) -> list:
    """
    Get n random CV splits [training, validation, and testing] or
    [training, validation] according to the ratios of split_frac.
    """
    a = list(a)
    # array indices to split on
    idx = [int(i) for i in np.cumsum(split_frac[:-1]) * len(a)]
    # for train, test, validation sets
    if len(split_frac) == 3:
        splits = []
        unique_tests = []
        unique_validations = []
        i = 0
        while len(splits) < n_splits:
            np.random.seed(i)
            np.random.shuffle(a)
            s = [list(ii) for ii in np.split(a, idx)]
            if set(s[1]) in unique_validations or set(s[2]) in unique_tests:
                continue
            unique_validations.append(set(s[1]))
            unique_tests.append(set(s[2]))
            splits.append({"train": s[0], "validation": s[1], "test": s[2]})
            i += 1
        return splits
    # for train, validation sets
    if len(split_frac) == 2:
        splits = []
        unique_validations = []
        i = 0
        while len(splits) < n_splits:
            np.random.seed(i)
            np.random.shuffle(a)
            s = [list(ii) for ii in np.split(a, idx)]
            if set(s[1]) in unique_validations:
                continue
            unique_validations.append(set(s[1]))
            splits.append({"train": s[0], "validation": s[1]})
            i += 1
        return splits


def get_loco_splits(
    df: pd.DataFrame,
    target: str,
    n_splits: int = 3,
    klims: tuple = (3, 10),
    n_folds: int = 10,
) -> list:
    """
    Get leave-one-cluster cross-validation splits using the
    algorithm described here: https://doi.org/10.1039/C8ME00012C.
    we choose random values of k in the klims range to create
    n_folds number of cluster splits, where each split has
    n_splits number of datasets.
    The splits are returned as a list using
    the dataframe index labels as indices.
    """

    # scale the input data
    dfscaled = standardize_df(df, ignore=[target])
    input_cols = get_num_cols(dfscaled, ignore=[target])

    ii = 0
    splits = []
    test_sets = []
    while len(splits) < n_folds:

        # get a random choice for k
        np.random.seed(ii)
        k = np.random.choice(np.arange(klims[0], klims[1] + 1))

        # shuffle data
        dfs = dfscaled.sample(frac=1, random_state=ii)
        dfs_index = dfs.index.tolist()

        # perform k-means clustering
        cluster = KMeans(
            n_clusters=k,
            init="random",
            random_state=k,
        ).fit(dfs[input_cols].values)
        clabels = list(cluster.labels_)

        # randomly assign each cluster to train, test, or validation partition
        partition_map = {
            label: np.random.randint(low=0, high=3) for label in np.unique(clabels)
        }

        # print(partition_map)
        split_lists = [[], [], []]
        for x in range(len(clabels)):
            partition_idx = partition_map[clabels[x]]
            split_lists[partition_idx].append(dfs_index[x])

        splits0 = {["train", "validation", "test"][i]: split_lists[i] for i in range(3)}

        # keep this split if certain conditions are met
        keep_split = all(
            [
                min([len(v) for _, v in splits0.items()]) >= 0.1 * len(dfs),
                set(splits0) not in test_sets,
            ]
        )

        if keep_split:
            splits.append(splits0)
            test_sets.append(set(splits0["test"]))
        ii += 1

    return splits


def get_cv_splits(ds: dict, klims: tuple = (3, 10)) -> dict:
    """Get cross-validation splits, using
    both LOCO and Random strategies."""
    # types of cross-validation to test
    cv_types = ["LOCO", "Random"]
    # save CV splits
    splits = {k: {} for k, _ in ds.items()}

    # loop over each dataset
    for di, dsn in enumerate(list(ds)):

        # get dataset
        print(f"Getting splits {di+1}: {dsn}")
        df = pd.DataFrame(json.loads(ds[dsn]["df"]))
        t = ds[dsn]["target"]

        # loop over CV strategies
        for ct in cv_types:
            # perform splitting
            if ct == "LOCO":
                s = get_loco_splits(df, target=t, n_splits=10, klims=klims)
                # perform mixing of the train and validation splits.
                # this is so the feature engineering (validation set)
                # doesn't have to go through 2 rounds of extrapolation -
                # first though feature engineering, then in model testing
                for s0 in s:
                    nontest = s0["train"] + s0["validation"]
                    s2 = get_random_splits(nontest, n_splits=1, split_frac=(0.5, 0.5))[
                        0
                    ]
                    s0["train"] = s2["train"]
                    s0["validation"] = s2["validation"]

            elif ct == "Random":
                s = get_random_splits(df.index, n_splits=10, split_frac=(0.8, 0.1, 0.1))
            splits[dsn][ct] = s
    return splits


def view_split_sizes(splits: dict):
    """View the number of samples in each split"""
    for dsn in splits:
        print(dsn)
        for split_type in splits[dsn]:
            print("  " + split_type)
            ss = splits[dsn][split_type]
            for ss0 in ss:
                print(
                    {k: len(v) for k, v in ss0.items()},
                    "total:",
                    sum([len(v) for _, v in ss0.items()]),
                )


@ignore_warnings()
def view_all_splits(
    cv_strategy: str, legend: bool = True, perplexity: int = 30, random_state: int = 0
):
    """View all train test splits of a given CV strategy"""

    # import the data
    ds = read_jsonzip(os.path.join("data", "datasets.json.gz"))
    splits = read_jsonzip(os.path.join("data", "splits.json.gz"))

    n_rows, n_cols = 9, 10
    offsets = 1.5, 1.5
    ax = plt.subplot(111)

    # loop over each dataset
    for di, dsn in enumerate(list(ds)):
        # get dataset
        print(f"{di+1}: {dsn}")
        df = pd.DataFrame(json.loads(ds[dsn]["df"]))

        t = ds[dsn]["target"]

        # get t-SNE embedding
        dfs = standardize_df(df[get_num_cols(df, ignore=t)])
        ts = TSNE(
            n_components=2, perplexity=perplexity, random_state=random_state
        ).fit_transform(dfs.values)
        ts = pd.DataFrame(columns=[1, 2], data=ts, index=dfs.index)
        ts[1] = norm_array(ts[1])
        ts[2] = norm_array(ts[2])

        for split_type in splits[dsn]:
            ss = splits[dsn][split_type]
            for si, s in enumerate(ss):

                if split_type == cv_strategy:

                    # get offsets for the plot
                    xoff, yoff = si * offsets[0], n_rows + 3 - (di * offsets[1])

                    ax.scatter(
                        ts.loc[s["train"], 1] + xoff,
                        ts.loc[s["train"], 2] + yoff,
                        c="green",
                        s=0.2,
                        lw=0,
                    )

                    ax.scatter(
                        ts.loc[s["validation"], 1] + xoff,
                        ts.loc[s["validation"], 2] + yoff,
                        c="mediumorchid",
                        s=0.2,
                        lw=0,
                    )

                    ax.scatter(
                        ts.loc[s["test"], 1] + xoff,
                        ts.loc[s["test"], 2] + yoff,
                        c="darkorange",
                        s=0.2,
                        lw=0,
                    )

                    if si == 0:
                        ax.text(
                            0,
                            yoff + 1,
                            t,
                            ha="left",
                            va="bottom",
                            fontsize=5,
                        )
    if legend:
        colors = ["green", "mediumorchid", "darkorange"]
        split_types = ["train", "validation", "test"]
        ax.legend(
            ncol=3,
            bbox_to_anchor=(0.44, 0.88),
            loc="lower center",
            # fontsize=16,
            handles=[
                mpatches.Patch(color=colors[i], alpha=0.75, label=split_types[i])
                for i in range(len(colors))
            ],
        )

    plt.setp(ax, "frame_on", False)
    ax.set_xlim([0, (n_cols + 1) * offsets[0]])
    ax.set_ylim([0, (n_rows + 1) * offsets[1]])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(
        os.path.join(FIG_BASEPATH, f"Splits{cv_strategy}.png"), bbox_inches="tight"
    )
    plt.show()


def plot_paper_figure_splits():
    """Plot example splits to use in paper schematic figure"""

    # import datasets and create CV splits
    ds = read_jsonzip(os.path.join("data", "datasets.json.gz"))
    ss = read_jsonzip(os.path.join("data", "splits.json.gz"))

    fig, ax = plt.subplots(nrows=4, ncols=2)
    ax = np.ravel(ax)
    ii = 0
    ALPHA = 1
    perplexity = 30
    dsn = "double_perovskites_gap"

    while ii < 8:

        df = pd.DataFrame(json.loads(ds[dsn]["df"]))
        t = ds[dsn]["target"]

        randd = 0
        if randd:
            ts = pd.DataFrame(
                columns=[1, 2], data=np.random.random((len(df), 2)), index=df.index
            )
        else:
            # get t-SNE embedding
            dfs = standardize_df(df[get_num_cols(df, ignore=t)])
            ts = TSNE(
                n_components=2, perplexity=perplexity, random_state=0
            ).fit_transform(dfs.values)
            ts = pd.DataFrame(columns=[1, 2], data=ts, index=df.index)

        ts[t] = df[t].values

        # divide data by split
        colors = ["green", "mediumorchid", "darkorange"]
        split_types = ["train", "validation", "test"]
        for color, split_type in zip(colors, split_types):

            cv_type = "Random" if ii < 4 else "LOCO"
            idx = ss[dsn][cv_type][ii][split_type]

            # main plot
            ax[ii].scatter(
                ts.loc[idx, 1],
                ts.loc[idx, 2],
                c=color,
                s=6,
                edgecolors="w",
                lw=0.1,
                alpha=ALPHA,
                label=split_type,
                zorder=ii,
            )
            ax[ii].scatter(
                ts.loc[ss[dsn][cv_type][ii]["train"], 1][::10],
                ts.loc[ss[dsn][cv_type][ii]["train"], 2][::10],
                c=["green"],
                s=6,
                edgecolors="w",
                lw=0.1,
                alpha=ALPHA,
                label=split_type,
                zorder=10,
            )

        ax[ii].set_xticks([])
        ax[ii].set_yticks([])
        for spine in ax[ii].spines.values():
            spine.set_edgecolor("white")
        # ax[ii].axis('off')

        ii += 1

    ax[0].text(
        1.3,
        1.05,
        "Interpolation (Random)",
        ha="center",
        fontsize=FONTSIZE + 1,
        transform=ax[0].transAxes,
    )
    ax[4].text(
        1.3,
        1.05,
        "Extrapolation (LOCO)",
        ha="center",
        fontsize=FONTSIZE + 1,
        transform=ax[4].transAxes,
    )

    if 1:
        ax[6].legend(
            ncol=3,
            bbox_to_anchor=(-0.3, -0.6),
            loc="lower left",
            fontsize=8,
            handles=[
                mpatches.Patch(color=colors[i], alpha=ALPHA, label=split_types[i])
                for i in range(len(colors))
            ],
        )

    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    fig.set_size_inches((2.5, 4))
    fig.savefig(
        os.path.join(FIG_BASEPATH, "SplitsPaper.png"),
        bbox_inches="tight",
    )
    plt.show()


def plot_target_vs_best_feature():
    """
    Create a grid pf scatter plots showing the target
    value vs the best engineerred feature value for each
    dataset and each LOCO split. SCatter points are colored
    by whether they came from the train, test, or validation sets.
    """
    ds = read_jsonzip(os.path.join("data", "datasets.json.gz"))
    splits = read_jsonzip(os.path.join("data", "splits.json.gz"))
    fs = read_jsonzip(os.path.join("data", "features.json.gz"))
    split_types = ["LOCO", "Random"]

    n_rows, n_cols = 9, 10
    offsets = 1.5, 1.5
    ax = plt.subplot(111)
    plt.setp(ax, "frame_on", False)
    ax.set_xlim([0, (n_cols + 1) * offsets[0]])
    ax.set_ylim([0, (n_rows + 1) * offsets[1]])
    ax.set_xticks([])
    ax.set_yticks([])

    # loop over each dataset
    for di, dsn in enumerate(list(ds)):
        # get dataset
        print(f"{di+1}: {dsn}")
        df = pd.DataFrame(json.loads(ds[dsn]["df"]))
        t = ds[dsn]["target"]

        # loop over each splitting strategy
        for split_type in split_types:
            ss = splits[dsn][split_type]

            # loop over each split
            for si, s in enumerate(ss):

                if split_type == "LOCO":

                    labels = []
                    for i, row in df.iterrows():
                        for k in ["train", "validation", "test"]:
                            if i in s[k]:
                                labels.append(k)

                    be_vals = json.loads(fs[dsn]["LOCO"][si])["val"]["0"]
                    df0 = pd.DataFrame(
                        {
                            "be_vals": be_vals,
                            t: df[t].values,
                        }
                    )
                    scaled_vals = MinMaxScaler().fit_transform(df0.values)
                    df0 = pd.DataFrame(
                        data=scaled_vals, columns=list(df0), index=list(df0.index)
                    )
                    df0["label"] = labels

                    train = df0[df0["label"] == "train"]
                    validate = df0[df0["label"] == "validation"]
                    test = df0[df0["label"] == "test"]

                    # get offsets for the plot
                    # xoff, yoff = si*offsets[0], di*offsets[1]
                    xoff, yoff = si * offsets[0], n_rows + 3 - (di * offsets[1])

                    ax.scatter(
                        train["be_vals"] + xoff, train[t] + yoff, c="green", s=0.4, lw=0
                    )
                    ax.scatter(
                        validate["be_vals"] + xoff,
                        validate[t] + yoff,
                        c="mediumorchid",
                        s=0.4,
                        lw=0,
                    )
                    ax.scatter(
                        test["be_vals"] + xoff,
                        test[t] + yoff,
                        c="darkorange",
                        s=0.4,
                        lw=0,
                    )

                    if si == 0:
                        ax.text(
                            0,
                            yoff + 1,
                            t,
                            ha="left",
                            va="bottom",
                            fontsize=5,
                        )

    if 1:
        colors = ["green", "mediumorchid", "darkorange"]
        split_types = ["train", "validation", "test"]
        ax.legend(
            ncol=3,
            bbox_to_anchor=(0.44, 0.88),
            loc="lower center",
            # fontsize=16,
            handles=[
                mpatches.Patch(color=colors[i], alpha=0.75, label=split_types[i])
                for i in range(len(colors))
            ],
        )

    plt.savefig(
        os.path.join(FIG_BASEPATH, "FeaturesVsTargets.png"), bbox_inches="tight"
    )
    plt.show()


def get_top_vars(cv: list = ["LOCO", "Random"], n: int = 10) -> dict:
    """ "
    Get the variables which were found for feature
    engineering with the highest freuencies.
    Returns a dict of {dataset: [variable]] key-val pairs.
    Examines only the specified cv strategy for finding
    top variables.
    """
    ds = read_jsonzip(os.path.join("data", "datasets.json.gz"))
    splits = read_jsonzip(os.path.join("data", "splits.json.gz"))
    fs = read_jsonzip(os.path.join("data", "features.json.gz"))
    # loop over each dataset
    topvars = {}
    for di, dsn in enumerate(list(ds)):
        df = pd.DataFrame(json.loads(ds[dsn]["df"]))
        t = ds[dsn]["target"]

        # get all the feature combos saved for this dataset
        feat_list = []
        for st in cv:
            for si in range(len(fs[dsn][st])):
                featdf = pd.DataFrame(json.loads(fs[dsn][st][si]))
                feat_list += [ff for fl in list(featdf["feat"]) for ff in fl]

        # get ranking of frequencies at which each variable shows up in engineered features
        var_freqs = pd.DataFrame(
            data=Counter(feat_list).most_common(), columns=["feat", "freq"]
        )
        topvars[dsn] = list(var_freqs["feat"])[:n]
    return topvars


def confidence_ellipse(x: list, y: list, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`, from example here:
    https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py.
    """
    x, y = np.array(x), np.array(y)
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    # used for looking at NDME scores
    good_points = [all([x[i] < 5, y[i] < 5]) for i in range(len(x))]
    x = x[good_points]
    y = y[good_points]

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # get eigenvalues of the 2D dataset
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # get the standard deviation of x from the sqrt of the variance
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.median(x)
    # get the standard deviation of y
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.median(y)
    # rotate, scale, and position the ellipse
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def get_all_model_results() -> pd.DataFrame:
    """Get all ML modeling results as a single dataframe"""
    # read result sets
    rs = pd.DataFrame(read_jsonzip(os.path.join("data", "results.json.gz")))
    # compile single dataframe with all results
    df = pd.DataFrame()
    for i, r0 in enumerate(rs["results"]):
        df0 = pd.DataFrame(r0)
        for kk in ["dataset", "model", "split_type", "input_type"]:
            df0[kk] = rs[kk][i]
        df = pd.concat([df, df0], ignore_index=True)
    return df


def add_additional_cols(
    df: pd.DataFrame,
    ignore: list = [],
    nat_log: bool = True,
    powers: list = [-4, -3, -2, -1, -0.5, -0.333, -0.25, 0.25, 0.33, 0.5, 2, 3, 4],
) -> pd.DataFrame:
    """
    Add additional columns to a dataframe by executing
    mathematical operations on existing columns.
    """
    # now raise existing columns to varying powers
    new_col_names = []
    new_vals_all = np.empty((len(df), 0))

    # loop over each column to use for creating additional columns
    for c in [cc for cc in df if cc not in ignore]:

        # get original column values
        vv = df[c].values

        # raise existing columns to various powers
        for p in powers:
            # first assess mathematical viability of different conditions.
            # for example, we can't perform (-2)^(1/2).
            # if conditions are met, save new column
            if any(
                [
                    p > 0 and isinstance(p, int),  # pos integer powers
                    p < 0 and isinstance(p, int) and 0 not in vv,  # neg integer powers
                    p > 0
                    and not isinstance(p, int)
                    and np.all(vv >= 0),  # pos non-int powers
                    p < 0
                    and not isinstance(p, int)
                    and np.all(vv > 0),  # neg non-int powers
                ]
            ):

                new_col_vals = np.float_power(vv, p).reshape((-1, 1))
                if (
                    not array_is_constant(new_col_vals)
                    and np.isfinite(new_col_vals).all()
                ):
                    new_vals_all = np.hstack((new_vals_all, new_col_vals))
                    new_col_names.append(f"{c}**{str(p)}")

        # take natural logs of existing columns
        if nat_log:
            if not c.startswith("ln ") and np.all(vv > 0):
                new_col_vals = np.log(vv).reshape((-1, 1))
                if (
                    not array_is_constant(new_col_vals)
                    and np.isfinite(new_col_vals).all()
                ):
                    new_vals_all = np.hstack((new_vals_all, new_col_vals))
                    new_col_names.append(f"ln {c}")

    # combine new columns with original columns in a single dataframe
    new_df = pd.DataFrame(data=new_vals_all, columns=new_col_names, index=df.index)
    df = pd.concat([df, new_df], axis=1)
    return df


def random_combos(
    inputs: list,
    n_combos: int = 1000,
    lims: tuple = (2, 5),
) -> list:
    """
    Get random combinations of inputs by selecting a
    number of combinations (n_combos), where each combination
    contains a random number of inputs in the range lims.
    """
    # choose a 2D random set of input indices.
    # we do twice the amount of n_combos initially to protect against
    # duplicates, which we remove, and then trim down to
    # n_combos.
    np.random.seed(0)
    idx_full = np.random.choice(
        np.arange(len(inputs)),
        size=(n_combos * 2, lims[1]),
    ).tolist()
    # choose the number of items to combine for each combo
    np.random.seed(0)
    n_inputs_to_choose = np.random.randint(
        low=lims[0],
        high=lims[1] + 1,
        size=n_combos * 2,
    )
    # prune the indices by the number of items to use
    idx = [idx_full[i][: n_inputs_to_choose[i]] for i in range(len(idx_full))]
    # create the combinations
    rc = [[inputs[i] for i in ii] for ii in idx]
    # sort the items in each combo so we can fillter duplicates
    for i in range(len(rc)):
        rc[i].sort()
    # remove duplicates
    return [list(item) for item in set(tuple(row) for row in rc)][:n_combos]


def vals_from_combos(df: pd.DataFrame, varlist: list):
    """
    Convert a list of column combinations into values
    based on the column values in a dataframe.
    """

    # for a single column
    if isinstance(varlist[0], str):
        vv = np.ones(len(df))

        # loop over each variable in the variable list
        for v in varlist:
            if len(v.split("**")) == 1:
                vv *= df[v].values
            else:
                v, power = v.split("**")
                vv *= df[v].values ** float(power)
        if not array_is_constant(vv) and np.isfinite(vv).all():
            return vv
        else:
            return None

    # for a 2D arrray
    else:
        vals = df.values
        cols = list(df)

        # loop over each list of variables
        for i in range(len(varlist)):

            # create new array of values
            vv = np.ones(len(vals))

            # loop over each variable in the variable list
            for v in varlist[i]:
                if len(v.split("**")) == 1:
                    vv *= df[v].values
                else:
                    v, power = v.split("**")
                    vv *= df[v] ** float(power)

            if not array_is_constant(vv) and np.isfinite(vv).all():
                vals = np.hstack((vals, vv.reshape((-1, 1))))
                cols.append(varlist[i])

        return cols, vals
