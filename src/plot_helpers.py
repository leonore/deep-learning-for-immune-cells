import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def show_image(img, title="untitled", cmap="gray", **kwargs):
    try:
        plt.imshow(img, cmap=cmap, **kwargs)
    except:
        plt.imshow(img[:, :, 0], cmap=cmap, **kwargs)
    plt.axis("off")
    plt.title(title)

def reshape(img, w=192, h=192, c=3):
    if c > 1:
      return np.reshape(img, (w, h, c))
    else:
      return np.reshape(img, (w, h))

def plot_range(imgs, RS=8):
  fig = plt.figure(figsize=(15, 15))
  for i in range(0, 5):
    ax = fig.add_subplot(1, 5, i)
    plt.imshow(imgs[i+rn])
    ax.axis('off')


def plot_clusters(X, y, labels=["Unstimulated", "OVA", "ConA"]):
    targets = range(len(labels))
    palette = np.array(sns.color_palette("hls", len(labels)))

    y = np.array(y)

    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot()

    for target, color, label in zip(targets, palette, labels):
        plt.scatter(X[y==target, 0], X[y==target, 1], c=[color], label=label)

    ax.axis('off')
    ax.grid(False)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plot_live(X, y, labels=["Unstimulated", "OVA", "ConA"]):
    targets = range(len(labels))
    palette = np.array(sns.color_palette("hls", len(labels)))

    annots = []
    axes = []
    scatters = []

    fig = plt.figure(figsize=(10,10))
    ax1 = plt.subplot()
    plt.axis('off')

for target, colour, label in zip(labels, palette, labels):
    ax2 = ax1.twinx()
    scatter2 = plt.scatter(tsne[y==target,0], tsne[y==target, 1], c=[colour], s=10, label=label)
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

        label = int(point.get_label())
        pos = point.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format(label)
        annot.set_text(text)


        arr_img = x_test[y==label][ind["ind"][0]]
        imagebox = OffsetImage(arr_img)
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
