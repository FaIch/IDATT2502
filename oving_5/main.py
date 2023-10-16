import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'data/agaricus-lepiota.data'
column_names = [
    'class',
    'cap-shape',
    'cap-surface',
    'cap-color',
    'bruises',
    'odor',
    'gill-attachment',
    'gill-spacing',
    'gill-size',
    'gill-color',
    'stalk-shape',
    'stalk-root',
    'stalk-surface-above-ring',
    'stalk-surface-below-ring',
    'stalk-color-above-ring',
    'stalk-color-below-ring',
    'veil-type',
    'veil-color',
    'ring-number',
    'ring-type',
    'spore-print-color',
    'population',
    'habitat'
]

df = pd.read_csv(file_path, header=None, names=column_names)

plt.figure(figsize=(10,6))
sns.countplot(x='habitat', hue='class', data=df, palette={'e': 'b', 'p': 'r'})
plt.title('Distribution of Edible and Poisonous Mushrooms by Habitat')
plt.xlabel('Habitat')
plt.ylabel('Count')
plt.legend(title='Class', labels=['Edible', 'Poisonous'])
plt.show()

df_dummies = pd.get_dummies(df, drop_first=True)

plt.spy(df_dummies, markersize=0.5)
fig = plt.gcf()
fig.set_size_inches(60,500)
plt.show()