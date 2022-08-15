import umap.umap_ as umap
from sklearn.manifold import TSNE
from datetime import datetime

from config import RS

def tsne_fn(x, y=None, random_state=RS):
    start = datetime.now()
    print("t-sne dimensionality reduction started at {}".format(start.strftime("%H:%M:%S")))
    x_tsne = TSNE(random_state=RS).fit_transform(x)
    print("t-sne took {} to finish".format(datetime.now() - start))
    return x_tsne

def umap_fn(x, y=None, random_state=RS, **kwargs):
    # WARNING: y shouldn't actually be passed in unless
    # for supervised clustering purposes
    start = datetime.now()
    print("UMAP dimensionality reduction started at {}".format(start.strftime("%H:%M:%S")))
    x_umap = umap.UMAP(random_state=RS, **kwargs).fit_transform(x, y)
    print("UMAP took {} to finish".format(datetime.now() - start))
    return x_umap

def run_both(x, y=None, random_state=RS):
    return tsne_fn(x, y, random_state), umap_fn(x, y, random_state)
