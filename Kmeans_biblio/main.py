from pandas import read_csv
import kmeans
df = read_csv('/Users/eliot/Desktop/Mall_Customers.csv')
#df = read_csv('/Users/eliot/Desktop/ushape.csv',header=None)

k = 5
columns = ['Annual Income (k$)','Spending Score (1-100)']
#columns = [0,1]

X = df[columns]


km = kmeans.Kmeans(nb_test=9,k=k,X=X,x_column=columns[0],y_column=columns[1])
km.model(df)
km.plot_clusters(df)

