import pandas as pd
from matplotlib import pyplot as plt

df_1 = pd.read_csv('models/incep_4_10-153503.csv')
df_2 = pd.read_csv('models/anet_4_10-163920.csv')

plt.plot(df_1['epoch'], df_1['accuracy'],label='Training Acc.')
plt.plot(df_1['epoch'], df_1['val_accuracy'], label='Validation Acc.')
plt.ylim((0.1))
plt.yticks([i * 0.1 for i in range(0, 11)])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('plots/incep4_acc.png')
plt.show()
plt.clf()

print(df_1['val_accuracy'])

plt.plot(df_2['epoch'], df_2['accuracy'],label='Training Acc.')
plt.plot(df_2['epoch'], df_2['val_accuracy'], label='Validation Acc.')
plt.ylim((0.1))
plt.yticks([i * 0.1 for i in range(0, 11)])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('plots/anet4_acc.png')
plt.show()