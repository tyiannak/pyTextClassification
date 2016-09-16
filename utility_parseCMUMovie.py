import os
import csv
import ast

# used to generate folder-seperated corpus from CMUMovie dataset 
# just type python utility_parseCMUMovie.py in a terminal and the data will be downloaded and split to subfolders in the moviePlots/ path

os.system("wget http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz")
os.system("tar -xvzf MovieSummaries.tar.gz")

minRevenue = 20000000

movieMetadata = {}
with open('MovieSummaries/movie.metadata.tsv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    for row in reader:        
        rev = 0
        if len(row[4])>1:
            rev = int(row[4])
        if (minRevenue < 0) or ( (minRevenue > 0) and (rev>minRevenue) ):
            movieMetadata[row[0]] = {}
            movieMetadata[row[0]]['title'] = row[2]        
            movieMetadata[row[0]]['genres'] = ast.literal_eval(row[8]).values()
        print len(movieMetadata)

with open("MovieSummaries/plot_summaries.txt") as f:
    content = f.readlines()

for c in content:
    d = c.split("\t")
    id = d[0]
    plot = d[1]
    if id in movieMetadata:
        print id, movieMetadata[id]['title']
        for g in movieMetadata[id]['genres']:
            if not os.path.exists("moviePlots" + os.sep + g.replace("/","-")):
                os.makedirs("moviePlots" + os.sep + g.replace("/","-"))
            f = open("moviePlots" + os.sep + g.replace("/","-") + os.sep + id + "_" + movieMetadata[id]["title"].replace("/","-"), 'w')        
            f.write(plot)
            f.close()
