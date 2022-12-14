import math
from scipy.spatial import distance
import pandas as pd
from itertools import cycle, islice
from sklearn.neighbors import KNeighborsRegressor
import random
from numpy.random import permutation
from scipy.stats import linregress
import matplotlib.pyplot as plt

#url = 'https://www.kaggle.com/datasets/thedevastator/the-ultimate-netflix-tv-shows-and-movies-dataset'
#df = pd.read_html(url)
data = 'Best Movies netflix.csv'
df = pd.read_csv(data)
df2 = df.head(30)
df2['SCORE']=df2['SCORE'].astype(float)
df2['NUMBER_OF_VOTES']=df2['NUMBER_OF_VOTES'].astype(float)
df3 = df2.sort_values(['MAIN_GENRE'])
df4 = df2.sort_values(['RELEASE_YEAR'])
df.set_index('RELEASE_YEAR', inplace=True)
d = df4.describe()
print(d)
#a list by cycling through the colors you care about to match the length of your data.
my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k', 'c', 'm', 'w', 'bisque', 'tan']), None, len(df4)))
x = df4.RELEASE_YEAR
y = df4.SCORE
plt.scatter(x, y, color = my_colors)
df4.groupby('MAIN_GENRE')['SCORE'].plot()
plt.legend(loc=2, prop={'size': 5.2})
df4.groupby('MAIN_GENRE').agg({
    'SCORE' : max,
    'DURATION' : pd.Series.mean
})
df4.groupby('MAIN_GENRE').agg(
    score = pd.NamedAgg(column='SCORE', aggfunc=max),
    duration = pd.NamedAgg(column='DURATION', aggfunc=pd.Series.mean),
)
pd.pivot_table(df4.reset_index(),
               index='DURATION', columns='MAIN_GENRE', values='SCORE'
              ).plot(kind = 'bar', subplots=False)
plt.legend(loc=6, prop={'size': 5})

df4.groupby(['MAIN_GENRE']).sum().plot(kind='pie', y='SCORE', title='Score based on Genre', autopct='%1.0f%%', legend = False)
plt.show()



selected_movie = df4[df4["TITLE"] == "Inception"].iloc[0]
score_columns = ['RELEASE_YEAR', 'SCORE', 'NUMBER_OF_VOTES', 'DURATION']
def euclidean_distance(row):
    inner_value = 0
    for k in score_columns:
        inner_value += (row[k] - selected_movie[k]) ** 2
    return math.sqrt(inner_value)

inception_distance = df4.apply(euclidean_distance, axis=1)
df4.fillna(0, inplace=True)
Inception = df4[df4["TITLE"] == "Inception"]
euclidean_distances = df4.apply(lambda row: distance.euclidean(row, Inception), axis=1)

distance_frame = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
distance_frame.sort("dist", inplace=True)
second_smallest = distance_frame.iloc[1]["idx"]
most_similar_to_Inception = df4.loc[int(second_smallest)]["TITLE"]
random_indices = permutation(df4.index)
test_cutoff = math.floor(len(df4)/3)
test = df4.loc[random_indices[1:test_cutoff]]
train = df4.loc[random_indices[test_cutoff:]]
