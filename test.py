import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from ExtendDT import ext_dt
from orapMethod import *

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=7)
cls_dt = DecisionTreeClassifier(random_state=0)
cls_dt.fit(X_train, y_train)
# fig = plt.figure(figsize=(20,15))
# tree.plot_tree(cls_dt, filled=True, rounded=True, node_ids=True)
# plt.show()
ORAP_base_sequence = get_ORAP_sequence(cls_dt)
ORAP_base_sequence.ShowSequence()
remove_sequence = ORAP_base_sequence.removed
prune_step = 4
pruned_xdt = series_prune(cls_dt, remove_sequence[prune_step])
fig = plt.figure(figsize=(20,15))
tree.plot_tree(pruned_xdt.base_dt, filled=True, rounded=True, node_ids=True)
plt.show()