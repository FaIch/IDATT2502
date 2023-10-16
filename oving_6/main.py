from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/agaricus-lepiota.csv')

le = LabelEncoder()
y = le.fit_transform(data['edibility'])

x = pd.get_dummies(data.drop(columns=['edibility']))

print("X shape:", x.shape)
print("y shape:", y.shape)

skb = SelectKBest(chi2, k=5)
skb.fit(x, y)
x_new = skb.transform(x)

print("x_new shape:", x_new.shape)

selected = [x.columns[i] for i in skb.get_support(indices=True)]
print("Best features to decide edibility:", ", ".join(selected))



# Creating PC components
pca = PCA(n_components=min(x.shape))
x_pca = pca.fit_transform(x)

# Plotting the variance based on cumulative variance in principal components
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum())
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Variance Explained by Principal Components")
plt.grid(True)
plt.show()

feature_names = x.columns

num_pcs_to_inspect = 5

# Showing the most influential feature of the 5 first PC's
for i in range(num_pcs_to_inspect):
    pc = pca.components_[i, :]
    most_important_idx = np.abs(pc).argmax()
    most_important_feature = feature_names[most_important_idx]

    print(f"Principal Component {i + 1}:")
    print(f"- Most influential feature: {most_important_feature}")
    print(f"- Explained Variance: {pca.explained_variance_ratio_[i]:.4f}")
    print(f"- Cumulative Explained Variance: {pca.explained_variance_ratio_[:i + 1].sum():.4f}\n")
