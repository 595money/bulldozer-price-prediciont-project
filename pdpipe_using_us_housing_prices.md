---
jupyter:
  jupytext:
    formats: ipynb,py:percent,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np
import pdpipe as pdp
```

```python
# Load data
df = pd.read_csv('data/USA_Housing.csv')
df.info()
```

```python
df.isna().sum()
```

```python
# insert NaN to dataset
np.random.seed(42)
df_miss = df.mask(np.random.random(df.shape) < .1)
df_miss.isna().sum()
```

```python
# def home size function
def size(n):
    if n <= 4:
        return 'Small'
    elif 4 < n <= 6:
        return 'Medium'
    else:
        return 'Big'
df_miss['House_size'] = df_miss['Avg. Area Number of Rooms'].apply(size)
```

```python
# 1. Pipe One-Hot-Encode
pipeline = pdp.OneHotEncode('House_size')
```

```python
def price_tag(x):
    if x > 2:
        return 'keep'
    else:
        return 'drop'
# 2. Pipe Drop column 
pipeline += pdp.RowDrop({'Price': lambda x: x<= 250000})

# 3. Pipe Apply func by cols
pipeline += pdp.ApplyByCols('Price', price_tag, 'Price_tag', drop=False)
pipeline(df_miss)
# pipeline += pdp.ColDrop('Avg. Area House Age')
```

```python
df2 = pipeline(df_miss)
```