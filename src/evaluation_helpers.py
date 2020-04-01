import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

import cv2
import numpy as np
from sklearn.metrics import mean_squared_error

from config import RS, imw, imh, c, evaluation_path

# QUICK VIS

def show_image(img, title="untitled", cmap="gray", **kwargs):
    try:
        plt.imshow(img, cmap=cmap, **kwargs)
    except:
        plt.imshow(img[:, :, 0], cmap=cmap, **kwargs)
    plt.axis("off")
    plt.title(title)


def reshape(img, w=imw, h=imh, c=c):
    if c > 1:
        return np.reshape(img, (w, h, c))
    else:
        return np.reshape(img, (w, h))


def plot_range(imgs, RS=0):
    fig = plt.figure(figsize=(15, 15))
    for i in range(0, 5):
        ax = fig.add_subplot(1, 5, i + 1)
        plt.imshow(imgs[i + RS])
        ax.axis('off')


## EVALUATION: REGRESSION

def metrics_report(y_true, y_pred, y_labels, labels, tag=None):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    sd = np.std(y_pred)

    categories = np.unique(y_labels)

    if tag:
        with open(evaluation_path + "regression/" + tag + "_metrics.txt", "w") as file:
            file.write("MSE score: {0:.3f}\n".format(mse))
            file.write("RMSE score: {0:.3f}\n".format(rmse))
            file.write("SD: {0:.3f}\n".format(sd))
            file.write("\n")

            for idx, cat in enumerate(categories):
                file.write("Scores for {}\n".format(labels[idx]))
                file.write("RMSE score: {0:.3f}\n".format(
                    np.sqrt(mean_squared_error(y_true[y_labels == cat], y_pred[y_labels == cat]))))
                file.write("SD: {0:.3f}\n\n".format(
                    np.std(y_pred[y_labels == cat])))

    print(
        "MSE score: {0:.3f} -- this is the average square difference between true and predicted".format(mse))
    print(
        "RMSE score: {0:.3f} -- difference between T and P in DV unit".format(rmse))
    print("SD of predictions: {0:.3f}\n".format(sd))

    for idx, cat in enumerate(categories):
        print("Scores for {}".format(labels[idx]))
        print("RMSE score: {0:.3f}".format(np.sqrt(mean_squared_error(
            y_true[y_labels == cat], y_pred[y_labels == cat]))))
        print("SD: {0:.3f}\n".format(np.std(y_pred[y_labels == cat])))


def plot_predictions_histogram(y_true, y_pred, y, labels=["Unstimulated", "OVA", "ConA"], tag=None):
    true = []
    pred = []
    for i in np.unique(y):
        true.append(y_true[y == i].flatten())
        pred.append(y_pred[y == i].flatten())

    palette = np.array(sns.color_palette("hls", len(labels)))

    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(211)

    plt.hist(pred, bins=16,
             label=labels,
             color=palette)

    ax1.yaxis.set_label_position("right")
    ax1.set_ylabel("Predicted values")
    plt.title("Histograms for predicted vs true overlaps")
    plt.legend()

    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
    plt.hist(true, bins=16,
             label=labels,
             color=palette)
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("True values")

    plt.xlabel("Level of interaction (Area of overlap)")
    plt.tight_layout()

    if tag:
        plt.savefig(evaluation_path + "regression/" +
                    tag + "_histogram.png", dpi=300)

    plt.show()


def plot_lines_of_best_fit(y_true, y_pred, y, labels=["Unstimulated", "OVA", "ConA"], tag=None):
    true = []
    pred = []
    for i in np.unique(y):
        true.append(y_true[y == i].flatten())
        pred.append(y_pred[y == i].flatten())

    palette = np.array(sns.color_palette("hls", len(labels)))
    colors = np.array(sns.hls_palette(len(labels), l=.5, s=.95))
    fig = plt.figure(figsize=(15, 5))

    for idx in range(len(labels)):
        s1 = plt.subplot(131 + idx, aspect='equal')
        s1.scatter(true[idx], pred[idx], c=[palette[idx]],
                   alpha=0.65, lw=0.1, edgecolor='k')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        s1.plot(true[idx], true[idx], '-', c=colors[idx])
        plt.title("Predictions - {} label".format(labels[idx]))

    plt.tight_layout()

    if tag:
        plt.savefig(evaluation_path + "regression/" +
                    tag + "_scatter.png", dpi=300)

    plt.show()


def plot_error_distribution(y_true, y_pred, tag=None):
    error = y_pred - y_true
    plt.hist(error.flatten(), bins=32)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title("Error distribution")

    if tag:
        plt.savefig(evaluation_path + "regression/" +
                    tag + "_error.png", dpi=300)

    plt.show()


## EVALUATION: AUTOENCODER

def plot_reconstruction(model, x_test, tag):
    plt.rcParams.update({'axes.titlesize': 'medium'})
    test_nb = np.random.randint(0, len(x_test) - 1)

    # show the difference in reconstruction
    decoded_imgs = model.predict(x_test[test_nb:test_nb + 1])

    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 2, 1)
    show_image(reshape(x_test[test_nb:test_nb + 1], w=imw,
                       h=imh, c=c), "original image [index {}]".format(test_nb))
    fig.add_subplot(1, 2, 2)
    show_image(reshape(decoded_imgs[0], w=imw,
                       h=imh, c=c), "reconstructed image")

    plt.tight_layout()

    if tag:
        plt.savefig(evaluation_path + "autoencoder/" +
                    tag + "_reconstruction.png", dpi=300)

    plt.show()


def plot_clusters(X, y, labels=["Unstimulated", "OVA", "ConA", "Faulty"], tag=None):
    targets = np.unique(y)
    palette = np.array(sns.color_palette("hls", len(labels)))

    y = np.array(y)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()

    for target, color, label in zip(targets, palette, labels):
        plt.scatter(X[y == target, 0], X[y == target, 1], c=[color],
                    label=label, alpha=0.8, s=25, edgecolor='k', lw=0.2)

    ax.axis('off')
    ax.grid(False)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-large')

    if tag:
        plt.savefig(evaluation_path + "clustering/" + tag + ".png", dpi=300)

    plt.show()


def plot_live(X, y, data, labels=["Unstimulated", "OVA", "ConA", "Faulty"]):
    """
    Plotting function for live visualisation of UMAP graphs
    A user can hover over point to see which image
    corresponds to which point
    /!\ can be slow for large and heavy datasets

    DISCLAIMER: this code was adapted from https://stackoverflow.com/a/58058600
    """
    targets = np.unique(y)
    palette = np.array(sns.color_palette("hls", len(labels)))

    annots = []
    axes = []
    scatters = []

    labels_dict = {}
    for idx, label in enumerate(labels):
        labels_dict[str(label)] = idx

    fig = plt.figure()
    ax1 = plt.subplot()
    plt.axis('off')
    plt.title("Live visualisation")

    for target, colour, label in zip(targets, palette, labels):
        ax2 = ax1.twinx()
        scatter2 = plt.scatter(X[y == target, 0], X[y == target, 1], c=[
                               colour], label=label, alpha=0.8, s=10, edgecolor='k', lw=0.2)
        ax2.grid(False)
        annot = ax2.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                             bbox=dict(boxstyle="round", fc="w", alpha=0.4),
                             arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        annots.append(annot)
        axes.append(ax2)
        scatters.append(scatter2)

        ax2.axis('off')
        ax2.grid(False)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(handles=scatters, loc='center left', bbox_to_anchor=(
        1, 0.5), labels=[str(x) for x in labels])

    annot_dic = dict(zip(axes, annots))
    line_dic = dict(zip(axes, scatters))

    def update_annot(point, annot, ind):

        label = labels_dict[point.get_label()]
        pos = point.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format(label)
        annot.set_text(text)

        arr_img = data[y == label][ind["ind"][0]]
        imagebox = OffsetImage(cv2.resize(
            arr_img, dsize=(96, 96), interpolation=cv2.INTER_AREA))
        imagebox.image.axes = ax1
        box = ax1.get_position()

        ab = AnnotationBbox(imagebox, xy=(1, 0),
                            xybox=(25, 10),
                            xycoords='axes fraction',
                            boxcoords="offset points",
                            bboxprops=dict(
                                alpha=0.2, linewidth=0.5, boxstyle="round"),
                            frameon=True)

        ax1.add_artist(ab)

    def hover(event):

        if event.inaxes in axes:
            for ax in axes:
                cont, ind = line_dic[ax].contains(event)
                annot = annot_dic[ax]
                if cont:
                    update_annot(line_dic[ax], annot, ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if annot.get_visible():
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()
