import umap.umap_ as UMAP
from sklearn.manifold import TSNE
from datetime import datetime

RS = 2211

def tsne(x, y=None, random_state=RS):
    start = datetime.now().strftime("%H:%M:%S")
    print("t-sne dimensionality reduction started at {}".format(start))
    x_tsne = TSNE(random_state=RS).fit_transform(x)
    print("t-sne took {} to finish".format(datetime.now() - start))
    return x_tsne

def umap(x, y=None, random_state=RS):
    # WARNING: y shouldn't actually be passed in unless
    # for supervised clustering purposes 
    start = datetime.now().strftime("%H:%M:%S")
    print("UMAP dimensionality reduction started at {}".format(start))
    x_umap = UMAP(random_state=RS).fit_transform(x, y)
    print("UMAP took {} to finish".format(datetime.now() - start))
    return x_umap

def run_both(x, y=None, random_state=RS):
    return tsne(x, y, random_state), umap(x, y, random_state)
