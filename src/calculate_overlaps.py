
# coding: utf-8

# In[57]:


import xlrd
import numpy as np
from scipy.ndimage import label

from dataset_helpers import minmax, reconstruct_from, get_label
from segmentation import get_mask, iou, threshold
from dataset_helpers import is_faulty

from plot_helpers import show_image
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# READ FILES
print("Reading input file")
npzfile = np.load("/Volumes/TARDIS/CK19_full.npz")
x = npzfile['x'] # images
y = npzfile['y']


# In[333]:


targets = np.ndarray(shape=(len(x)//2, 1), dtype=np.float32)
calculated = np.ndarray(shape=(len(x)//2, 1), dtype=np.float32)
labels = np.ndarray(shape=(len(y)//2), dtype=np.float32)


# In[168]:


book = xlrd.open_workbook("../data/target/CK19_metrics.xlsx")
sheet = book.sheet_by_index(0)
target_dict = {}
for i in range(1, sheet.nrows):
    target_dict[sheet.row_values(i)[0]] = sheet.row_values(i)[1]


# In[173]:


assert len(target_dict) == len(x)//2/100


# In[334]:


# initialise index values
idx = 0
i = 0
count = 0

print("Looping through images...")
while idx < len(x)-100:
    # ignore 100, 300, etc. values as they will already have been processed
    if count == 100:
        count = 0
        idx += 100
    else:
        if is_faulty(x[idx]) or is_faulty(x[idx+100]):
            overlap = 0
        else:
            tcell = get_mask(minmax(x[idx].astype(np.float32)))
            dcell = get_mask(minmax(x[idx+100].astype(np.float32)))
            overlap = iou(tcell,dcell)

        file = y[idx].split("/")[-1].split("(")[0]
        targets[i] = target_dict.get(file)
        labels[i] = get_label(y[idx])
        calculated[i] = overlap
        
        i += 1
        idx += 1
        count += 1

print('Overlaps have been counted')


# In[337]:


assert len(targets) == len(labels) == len(calculated)


# In[340]:


palette = np.array(sns.color_palette("hls", 4))[:3]


# In[362]:


unstimulated = np.unique(targets[labels==0])
ova = np.unique(targets[labels==1])
cona = np.unique(targets[labels==2])


# In[359]:


plt.figure(figsize=(10,5))
plt.hist([unstimulated, ova, cona], bins=64, histtype="barstacked",
         label=["Unstimulated", "OVA", "ConA"],
         color=palette)
plt.title("Histogram for target overlaps")
plt.legend()
plt.xlabel("Level of interaction (Mean % of overlap)")
plt.show()


# In[364]:


unstimulated = calculated[labels==0]
ova = calculated[labels==1]
cona = calculated[labels==2]


# In[366]:


plt.figure(figsize=(10,5))
plt.hist([unstimulated.flatten(), ova.flatten(), cona.flatten()], bins=64, histtype="barstacked", 
         label=["Unstimulated", "OVA", "ConA"],
         color=palette)
plt.title("Histogram for calculated overlaps")
plt.legend()
plt.xlabel("Level of interaction (Area of overlap)")
plt.show()


# In[367]:


unstimulated = [np.sum(unstimulated[current: current+100]) for current in range(0, len(unstimulated), 100)]
ova = [np.sum(ova[current: current+100]) for current in range(0, len(ova), 100)]
cona = [np.sum(cona[current: current+100]) for current in range(0, len(cona), 100)]


# In[369]:


plt.figure(figsize=(10,5))
plt.hist([unstimulated, ova, cona], bins=64, histtype="barstacked", 
         label=["Unstimulated", "OVA", "ConA"],
        color=palette)
plt.title("Histogram for calculated overlaps (summed over same images)")
plt.legend()
plt.xlabel("Level of interaction (Sum area of overlap per image)")
plt.show()


# In[374]:


# initialise index values
idx = 0
i = 0
count = 0

print("Looping through images to put faulty labels in...")
while idx < len(x)-100:
    # ignore 100, 300, etc. values as they will already have been processed
    if count == 100:
        count = 0
        idx += 100
    else:
        if is_faulty(x[idx]) or is_faulty(x[idx+100]):
            overlap = 0
            labels[i] = 3
        i += 1
        idx += 1
        count += 1

print('Faulty labels entered into array.')


# In[379]:


np.savez("../data/processed/CK19_calculated_metrics.npz", targets=targets, calculated=calculated, labels=labels)

