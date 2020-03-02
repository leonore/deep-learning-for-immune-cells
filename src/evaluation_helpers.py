import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

import cv2
import numpy as np
from sklearn.metrics import mean_squared_error

from config import RS, imw, imh, c, evaluation_path

## QUICK VIS

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

def plot_range(imgs, RS=RS):
  fig = plt.figure(figsize=(15, 15))
  for i in range(0, 5):
    ax = fig.add_subplot(1, 5, i)
    plt.imshow(imgs[i+rn])
    ax.axis('off')

## EVALUATION

# CREDIT: https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34019
def r2_score(y_true, y_pred):
    ssres = np.sum(np.square(y_true - y_pred))
    sstot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - ssres / sstot

def metrics_report(y_true, y_pred, tag=None):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    if tag:
        with open(repo_path + "data/evaluation/regression/" + tag + "_metrics.txt", "w") as file:
            file.write("MSE score: {}".format(mse))
            file.write("RMSE score: {}".format(rmse))
            file.write("R2 score: {}".format(r2))
    else:
        print("MSE score: {} -- this is the average square difference between true and predicted".format(mse))
        print("RMSE score: {} -- difference between T and P in DV unit".format(rmse))
        print("R2 score: {} -- explains variance. closest to 1 is better".format(r2))


def plot_clusters(X, y, labels=["Unstimulated", "OVA", "ConA", "Faulty"], tag=None):
    targets = range(len(labels))
    palette = np.array(sns.color_palette("hls", len(labels)))

    y = np.array(y)

    fig = plt.figure()
    ax = plt.subplot()

    for target, color, label in zip(targets, palette, labels):
        plt.scatter(X[y==target, 0], X[y==target, 1], c=[color], label=label, alpha=0.60, s=10, edgecolor='k', lw=0.2)

    ax.axis('off')
    ax.grid(False)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

    if tag:
        plt.savefig(evaluation_path + "clustering/" + tag + ".png")


def plot_predictions_histogram(x_true, x_pred, y, tag=None):
    t_u = x_true[y==0].flatten() # true unstimulated
    t_o = x_true[y==1].flatten() # true ova
    t_c = x_true[y==2].flatten() # true cona

    p_u = x_pred[y==0].flatten() # predicted unstimulated
    p_o = x_pred[y==1].flatten() # predicted ova
    p_c = x_pred[y==2].flatten() # predicted cona

    palette = np.array(sns.color_palette("hls", 4))[:3]

    fig = plt.figure(figsize=(10,5))
    ax1 = plt.subplot(211)

    plt.hist([p_u, p_o, p_c], bins=32,
         label=["Unstimulated", "OVA", "ConA"],
         color=palette)

    ax1.yaxis.set_label_position("right")
    ax1.set_ylabel("Predicted values")
    plt.title("Histograms for predicted vs true overlaps")
    plt.legend()

    ax2=plt.subplot(212, sharex=ax1, sharey=ax1)
    plt.hist([t_u, t_o, t_c], bins=32,
         label=["Unstimulated", "OVA", "ConA"],
         color=palette)
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("True values")

    plt.xlabel("Level of interaction (Area of overlap)")
    plt.tight_layout()
    plt.show()

    if tag:
        plt.savefig(evaluation_path + "/regression/" + tag + "_histogram.png")

def plot_lines_of_best_fit(x_true, x_pred, y, tag=None):
    t_u = x_true[y==0].flatten() # true unstimulated
    t_o = x_true[y==1].flatten() # true ova
    t_c = x_true[y==2].flatten() # true cona

    p_u = x_pred[y==0].flatten() # predicted unstimulated
    p_o = x_pred[y==1].flatten() # predicted ova
    p_c = x_pred[y==2].flatten() # predicted cona

    palette = np.array(sns.color_palette("hls", 4))[:3]

    fig = plt.figure(figsize=(16,16))
    s1 = plt.subplot(131, aspect='equal')
    s1.scatter(t_u, p_u, c=[palette[0]], alpha=0.65, lw=0.1, edgecolor='k')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    s1.plot(t_u, t_u, 'red')
    plt.title("Predictions - Unstimulated label")

    s2 = plt.subplot(132, aspect='equal')
    s2.scatter(t_o, p_o, c=[palette[1]], alpha=0.65, lw=0.1, edgecolor='k')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    s2.plot(t_o, t_o, 'green')
    plt.title("Predictions - OVA label")

    s3 = plt.subplot(133, aspect='equal')
    s3.scatter(t_c, p_c, c=[palette[2]], alpha=0.65, lw=0.1, edgecolor='k')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    s3.plot(t_c, t_c, 'blue')
    plt.title("Predictions - ConA label")

    plt.tight_layout()
    plt.show()

    if tag:
        plt.savefig(evaluation_path + "regression/" + tag + "_scatter.png")


def plot_error_distribution(y_true, y_pred, tag=None):
    error = y_pred - y_true
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title("Error distribution")
    plt.show()

    if tag:
        plt.savefig(evaluation_path + "regression/" + tag + "_error.png")


def plot_live(X, y, data, labels=["Unstimulated", "OVA", "ConA", "Faulty"]):
    targets = range(len(labels))
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

    for target, colour, label in zip(targets, palette, labels):
        ax2 = ax1.twinx()
        scatter2 = plt.scatter(X[y==target,0], X[y==target, 1], c=[colour], s=10, label=label, alpha=0.75)
        ax2.grid(False)
        annot = ax2.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
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
    ax1.legend(handles=scatters, loc='center left', bbox_to_anchor=(1, 0.5), labels=[str(x) for x in labels])

    annot_dic = dict(zip(axes, annots))
    line_dic = dict(zip(axes, scatters))

    def update_annot(point, annot, ind):

        label = labels_dict[point.get_label()]
        pos = point.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format(label)
        annot.set_text(text)


        arr_img = data[y==label][ind["ind"][0]]
        imagebox = OffsetImage(cv2.resize(arr_img, dsize=(48,48), interpolation=cv2.INTER_NEAREST))
        imagebox.image.axes = ax1
        box = ax1.get_position()

        ab = AnnotationBbox(imagebox, xy=(1, 0),
                            xybox=(25, 10),
                            xycoords='axes fraction',
                            boxcoords="offset points",
                            bboxprops=dict(alpha=0.2, linewidth=0.5, boxstyle="round"),
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
