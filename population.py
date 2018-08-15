from sklearn import linear_model
import matplotlib.pyplot as plt
features = [[2012],[2013],[2014],[2015],[2016],[2017]]
labels = [1233,1249,1266,1282,1299,1334]
plt.scatter(features, labels, color='red')
plt.xlabel('population')
plt.ylabel('year')

clf = linear_model.LinearRegression()
clf = clf.fit(features, labels)
result = clf.predict([[2012], [2013], [2014], [2015], [2016], [2017], [2018], [2019]])

plt.plot([[2012], [2013], [2014], [2015], [2016], [2017], [2018], [2019]], result, color='blue', linewidth=3)

plt.show()
