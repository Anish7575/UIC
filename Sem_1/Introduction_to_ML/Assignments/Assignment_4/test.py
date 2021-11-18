import numpy as np
import pandas as pd

df1 = pd.DataFrame({'c1':['rfgedf'], 'c2':[2], 'c3':[3]})
df1 = df1.set_index(pd.Index([5]))
df2 = pd.DataFrame({'c1':['fwrf'], 'c2':[5], 'c3':[6]})
df2 = df2.set_index(pd.Index([5]))

l = []
l.append(df1)
l.append(df2)

print(l)
print(pd.concat(l, ignore_index=True))
