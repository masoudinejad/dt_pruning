import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from ExtendDT import ext_dt
from orapMethod import get_ORAP_sequence

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=7)
cls_dt = DecisionTreeClassifier(random_state=0)
cls_dt.fit(X_train, y_train)
ORAP_base_sequence = get_ORAP_sequence(cls_dt)
ORAP_base_sequence.ShowSequence()