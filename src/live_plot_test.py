import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np; np.random.seed(11)

from keras.datasets import mnist
from sklearn.manifold import TSNE

(_, _), (x_test, y_test) = mnist.load_data()

tsne = TSNE().fit_transform(x_test[:1000].reshape(1000, 28*28))
y = y_test[:1000]

annots = []
axes = []
scatters = []

labels = [x for x in range(10)]
palette = np.array(sns.color_palette("hls", len(labels)))

fig = plt.figure()
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
    pos = point.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}".format(" ".join(list(map(str,ind["ind"]))))
    annot.set_text(text)

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
