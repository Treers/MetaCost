## MetaCost
This method was first proposed by P. Domingos, and I coded it with Python.

<p align="center">
<img src="https://github.com/Treers/MetaCost/blob/master/etc/metacost.jpg" />
</p>

## Example
Below is an example of using MetaCost to perform imbalanced learning.
```python
 >>> from sklearn.datasets import load_iris
 >>> from sklearn.linear_model import LogisticRegression
 >>> import pandas as pd
 >>> import numpy as np
 >>> S = pd.DataFrame(load_iris().data)
 >>> S['target'] = load_iris().target
 >>> LR = LogisticRegression(solver='lbfgs', multi_class='multinomial')
 >>> C = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
 >>> model = MetaCost(S, LR, C).fit('target', 3)
 >>> model.predict_proba(load_iris().data[[2]])
 >>> model.score(S[[0, 1, 2, 3]].values, S['target'])
```

Reference
---------
[P. Domingos, "MetaCost: A General Method for Making Classifiers Cost-Sensitive", Proc. Int’l Conf. Knowledge Discovery and Data Mining, pp. 155-164, 1999.](https://homes.cs.washington.edu/~pedrod/papers/kdd99.pdf)
