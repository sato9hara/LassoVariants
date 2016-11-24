# AlternateLasso
Python code for the alternate features search proposed in the following paper.

* S. Hara, T. Maehara, [Finding Alternate Features in Lasso](https://arxiv.org/abs/1611.05940), arXiv:1611.05940, 2016.

## Requirements
To use AlternateLasso:

* Python3.x
* Numpy
* Scikit-learn

## Usage

Prepare data:

* Input ``X``: feature matrix, numpy array of size (num, dim).
* Output ``y``: output array, numpy array of size (num,).
  * For regression, ``y`` is real value.
  * For classification, ``y`` is binary class index (i.e., 0 or 1).

Import the class:

```python
# for regression
from AlternateLinearModel import AlternateLasso
# for classification
from AlternateLinearModel import AlternateLogisticLasso
```

Fit the model:

```python
mdl = AlternateLasso(rho=0.1, verbose=True) # for regression
#mdl = AlternateLogisticLasso(rho=0.1, verbose=True) # for classification
mdl.fit(X, y)
```

Check the found alternate features:

```python
print(mdl)
```

For further deitals, see ``AlternateLinearModel.py``.
In IPython, one can check:

```python
import AlternateLinearModel
AlternateLinearModel?
```

## Examples

### Example 1 - Replicating Paper Results

In ``paper`` directory:

```
python paper_news20_task1.py > news20_task1.txt
python paper_news20_task2.py > news20_task2.txt
```

### Example 2 - Synthetic Data Experiment

In ``example`` directory:

```
python example_regression.py
python example_classification.py
```

The result of ``python example_regression.py`` would be someting like this:

```
> [feature name, # of alternate feature candidates]
> [x_1, 2]
> [x_2, 0]

Feature: x_1, Coef. = 0.907215
         Alternate Feature: x_3, Score = 0.081963, Coef. = 0.720439
         Alternate Feature: x_4, Score = 0.081587, Coef. = 0.729658
Feature: x_2, Coef. = 0.212481
         ** No Alternate Features **
```
