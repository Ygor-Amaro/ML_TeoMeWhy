# %%
import pandas as pd
data = "data/dados_cerveja.xlsx"

df = pd.read_excel(data)
df.head()

# %% 

features = ['temperatura', 'copo', 'espuma', 'cor']
target = 'classe'

X = df[features]
y = df[target]

"""
o scikit-learn só consegue trabalhar com váriaveis numéricas
então valores strings precisam ser alteradas para número
"""
X = X.replace(
    {
        "mud": 1, "pint": 2,
        "sim": 1, "não": 0,
        "clara": 0, "escura": 1,
    }
)

print(X.head())
print(y.head())

# %%
from sklearn import tree

seed = 42
model = tree.DecisionTreeClassifier()

model.fit(X=X, y=y)

# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model,
               feature_names=features,
               class_names=model.classes_,
               filled=True
               )

# %%
