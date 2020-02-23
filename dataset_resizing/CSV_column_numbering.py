import numpy as np
import pandas as pd

data = pd.read_csv("/home/onu/PycharmProjects/Heart/train/features.csv")



df = pd.DataFrame(data)

export_csv = df.to_csv ('1.csv', index = False, header=True) #Don't forget to add '.csv' at the end of the path
