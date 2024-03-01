#%%
from surroundmodulation.utils.misc import pickleread, picklesave


d0 = pickleread('/project/experiment_data/convnext/data_v3_0.pickle')
d1 = pickleread('/project/experiment_data/convnext/data_v3_1.pickle')
d2 = pickleread('/project/experiment_data/convnext/data_v3_2.pickle')
d3 = pickleread('/project/experiment_data/convnext/data_v3_3.pickle')

d0.update(d1)
d0.update(d2)
d0.update(d3)
picklesave('/project/experiment_data/convnext/data_v3.pickle', d1)

# %%
