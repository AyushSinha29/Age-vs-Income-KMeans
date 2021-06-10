import pandas as pd
df = pd.read_csv("/content/K MEANS2.csv")
df.columns = df.columns.str.strip()
import matplotlib.pyplot as plt
plt.scatter(df["Age"],df["Income($)"])
plt.show()
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2)
df.drop('Name',axis='columns', inplace=True)
y_pred = km.fit_predict(df)
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
plt.scatter(df1["Age"],df1["Income($)"], color="green")
plt.scatter(df2["Age"],df2["Income($)"], color="red")
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color= "purple", marker = "*", label="centroid")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
