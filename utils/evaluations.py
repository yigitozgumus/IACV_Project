import os
import csv
import numpy as np
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn import manifold
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, precision_recall_curve
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import time
import matplotlib.cm as cm
from utils.utils import working_directory
from sklearn.metrics.pairwise import euclidean_distances

# import cv2
sns.set(color_codes=True)


def do_prc(scores, true_labels, file_name="", directory="", plot=True):
    """ Does the PRC curve
    Args :
            scores (list): list of scores from the decision function
            true_labels (list): list of labels associated to the scores
            file_name (str): name of the PRC curve
            directory (str): directory to save the jpg file
            plot (bool): plots the PRC curve or not
    Returns:
            prc_auc (float): area under the under the PRC curve
    """
    precision, recall, thresholds = precision_recall_curve(true_labels, scores)
    prc_auc = auc(recall, precision)

    if plot:
        plt.figure()
        plt.step(recall, precision, color="b", alpha=0.2, where="post")
        plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title("Precision-Recall curve: AUC=%0.4f" % (prc_auc))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with working_directory(directory):
            plt.savefig(file_name + "_prc.jpg")
            plt.close()

    return prc_auc


def do_roc(scores, true_labels, file_name="", directory="", plot=True):
    """ Does the ROC curve

    Args:
            scores (list): list of scores from the decision function
            true_labels (list): list of labels associated to the scores
            file_name (str): name of the ROC curve
            directory (str): directory to save the jpg file
            plot (bool): plots the ROC curve or not
    Returns:
            roc_auc (float): area under the under the ROC curve
            thresholds (list): list of thresholds for the ROC
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    fpr, tpr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)  # compute area under the curve
    if plot:
        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve (area = %0.3f)" % (roc_auc))
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.legend(loc="lower right")

        plt.savefig(directory + file_name + "roc.png")
        plt.close()

    return roc_auc


def do_cumdist(scores, file_name="", directory="", plot=True):
    N = len(scores)
    X2 = np.sort(scores)
    F2 = np.array(range(N)) / float(N)
    if plot:
        plt.figure()
        plt.xlabel("Anomaly score")
        plt.ylabel("Percentage")
        plt.title("Cumulative distribution function of the anomaly score")
        plt.plot(X2, F2)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + file_name + "cum_dist.png")


def get_percentile(scores, dataset):
    if dataset == "material":
        # Highest 20% are anomalous
        per = np.percentile(scores, 94)
    return per


def do_hist(scores, true_labels, directory, dataset, random_seed, step=None, display=False):
    plt.figure()
    idx_inliers = true_labels == 0
    idx_outliers = true_labels == 1
    hrange = (min(scores), max(scores))
    plt.hist(
        scores[idx_inliers],
        50,
        facecolor=(0, 1, 0, 0.5),
        label="Normal samples",
        density=True,
        range=hrange,
    )
    plt.hist(
        scores[idx_outliers],
        50,
        facecolor=(1, 0, 0, 0.5),
        label="Anomalous samples",
        density=True,
        range=hrange,
    )
    plt.title("Distribution of the anomaly score")
    plt.legend()
    if display:
        plt.show()
    else:
        if step == None:
            plt.savefig(
                directory + "histogram_{}_{}.png".format(random_seed, dataset),
                transparent=True,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                directory + "histogram_{}_{}_{}.png".format(random_seed, dataset, step),
                transparent=True,
                bbox_inches="tight",
            )
        plt.close()


def do_hists(scores, true_labels, directory, dataset, random_seed, display=False):
    plt.figure()
    n_samples = len(scores)
    n_labels = np.max(true_labels)
    hrange = (min(scores), max(scores))
    for l in range(n_labels):
        idx = true_labels == l
        plt.hist(
            scores[idx],
            50,
            facecolor=(0, 1, 0, 0.5),
            label="{}".format(l),
            density=True,
            range=hrange,
        )
    plt.title("Distribution of the anomaly score")
    plt.legend()
    if display:
        plt.show()
    else:
        plt.savefig(
            directory + "hists_{}_{}.png".format(random_seed, dataset),
            transparent=True,
            bbox_inches="tight",
        )
        plt.close()


def predict(scores, threshold):
    return scores >= threshold


def make_meshgrid(x_min, x_max, y_min, y_max, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def save_grid_plot(samples, samples_rec, name_model, dataset, nb_images=50, grid_width=10):
    args = name_model.split("/")[:-1]
    directory = os.path.join(*args)
    if not os.path.exists(directory):
        os.makedirs(directory)
    samples = (samples + 1) / 2
    samples_rec = (samples_rec + 1) / 2

    figsize = (32, 32)
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(grid_width, grid_width)
    gs.update(wspace=0.05, hspace=0.05)
    list_samples = []
    for x, x_rec in zip(
        np.split(samples, nb_images // grid_width), np.split(samples_rec, nb_images // grid_width)
    ):
        list_samples += np.split(x, grid_width) + np.split(x_rec, grid_width)
    list_samples = [np.squeeze(sample) for sample in list_samples]
    for i, sample in enumerate(list_samples):
        if i >= nb_images * 2:
            break
        ax = plt.subplot(gs[i])
        if dataset == "mnist":
            plt.imshow(sample, cmap=cm.gray)
        else:
            plt.imshow(sample)
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
    plt.savefig("{}.png".format(name_model))


def save_results(
    location,
    scores,
    true_labels,
    model,
    dataset,
    method,
    weight,
    label,
    random_seed,
    logger,
    step=-1,
    percentile=90,
):
    directory = location + "{}/{}/{}/w{}/".format(model, dataset, method, weight)
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_name = "{}_step{}_rd{}".format(label, step, random_seed)
    fname = directory + "{}.csv".format(label)

    scores = np.array(scores)

    roc_auc = do_roc(scores, true_labels, file_name=file_name, directory=directory)
    prc_auc = do_prc(scores, true_labels, file_name=file_name, directory=directory)
    do_cumdist(scores, file_name=file_name, directory=directory)

    do_hist(scores, true_labels, directory, dataset, random_seed, None)
    if np.max(true_labels) > 1:
        do_hists(scores, true_labels, directory, dataset, random_seed)

    if type(percentile) == int:
        per = get_percentile(scores, dataset)
    else:
        per = np.percentile(scores, percentile)

    if type(per) == np.float64:
        y_pred = scores >= per

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels.astype(int), y_pred.astype(int), average="binary"
        )

        logger.info("Testing at step %i, method %s: AUROC = %.4f" % (step, method, roc_auc))

        results = [
            model,
            dataset,
            method,
            weight,
            label,
            step,
            per,
            roc_auc,
            prc_auc,
            precision,
            recall,
            f1,
            random_seed,
            time.ctime(),
        ]
        save_results_csv(location + "results.csv", results, header=0)

        results = [step, roc_auc, precision, recall, f1, random_seed]
        save_results_csv(fname, results, header=2)

    elif type(per) == np.ndarray:
        for i in per:
            y_pred = scores >= i

            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels.astype(int), y_pred.astype(int), average="binary"
            )

            logger.info(
                "Testing at step %i, with percentile %f, method %s: AUROC = %.4f"
                % (step, i, method, roc_auc)
            )

            results = [
                model,
                dataset,
                method,
                weight,
                label,
                step,
                i,
                roc_auc,
                prc_auc,
                precision,
                recall,
                f1,
                random_seed,
                time.ctime(),
            ]
            save_results_csv(location + "results.csv", results, header=0)

            results = [step, i, roc_auc, precision, recall, f1, random_seed]
            fname = directory + "{}.csv".format(label)
            save_results_csv(fname, results, header=2)


def heatmap(data, name=None, save=False):
    fig = plt.figure()
    ax = sns.heatmap(data, cmap="YlGnBu")
    fig.add_subplot(ax)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if save:
        args = name.split("/")[:-1]
        directory = os.path.join(*args)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig("{}.png".format(name))
    return data


def save_results_csv(fname, results, header=0):
    """Saves results in csv file
    Args:
        fname (str): name of the file
        results (list): list of prec, rec, F1, rds
    """

    new_rows = []
    if not os.path.isfile(fname):
        args = fname.split("/")[:-1]
        directory = os.path.join(*args)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(fname, "wt") as f:
            writer = csv.writer(f)
            if header == 0:
                writer.writerows(
                    [
                        [
                            "Model",
                            "Dataset",
                            "Method",
                            "Weight",
                            "Label",
                            "Step",
                            "Percentile",
                            "AUROC",
                            "PRC",
                            "Precision",
                            "Recall",
                            "F1 score",
                            "Random Seed",
                            "Date",
                        ]
                    ]
                )
            if header == 1:
                writer.writerows([["Precision", "Recall", "F1 score", "Random Seed"]])
            elif header == 2:
                writer.writerows(
                    [
                        [
                            "Step",
                            "Percentile",
                            "AUROC",
                            "Precision",
                            "Recall",
                            "F1 score",
                            "Random Seed",
                        ]
                    ]
                )

            elif header == 5:
                writer.writerows(
                    [["Model", "Dataset", "Method", "Weight", "Label", "Step", "Scores"]]
                )

    with open(fname, "at") as f:
        # Overwrite the old file with the modified rows
        writer = csv.writer(f)
        new_rows.append(results)  # add the modified rows
        writer.writerows(new_rows)

def determine_normality_param(train_codebook,valid_recon,alpha=1,m=200):

    K = euclidean_distances(valid_recon,train_codebook)
    indmat = np.argsort(K,axis=1)
    topm_K = []
    for i in range(K.shape[0]):
        idx = indmat[i,:m]
        topm_K.append(K[i,idx])
    topm_K = np.array(topm_K)
    d = np.mean(topm_K,axis=1)
    mu_th = np.mean(d)
    sigma_th = np.std(d)
    alpha = 1
    threshold = mu_th + sigma_th*alpha
    return threshold

def predict_anomaly(test_batch,train_codebook,threshold,m=200):    
    # test_batch = np.reshape(test_batch,[test_batch.shape[0],test_batch.shape[1]*test_batch.shape[2]])
    K = euclidean_distances(test_batch,train_codebook)
    indmat = np.argsort(K,axis=1)
    topm_K = []
    for i in range(K.shape[0]):
        idx = indmat[i,:m]
        topm_K.append(K[i,idx])
    topm_K = np.array(topm_K)
    d_test = np.mean(topm_K,axis=1)
    return threshold<d_test,d_test