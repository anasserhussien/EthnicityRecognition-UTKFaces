#%% 
import matplotlib.pyplot as plt
import os

UTK_DIST = 'data/utk_races'
CLASSES = ['White', 'Black', 'Asian', 'Indian']

# Visualize data distribution
data_dist = []
for i in range(len(CLASSES)):
    data_dist.append(len(os.listdir(f'{UTK_DIST}/test/{i}/')))

plt.bar(CLASSES, data_dist, width=0.5)
plt.xlabel('Ethnicities')
plt.ylabel('# of instances')
plt.title('Testing data distribution')
plt.savefig('visualizations/test_dist.png')
plt.show()
# %%