"""
prepare matrix of contrasts by concepts
"""

import pandas
import numpy
from pyneurovault import api

# Read in contrast and image list
images = pandas.read_csv("contrast_data.tsv",sep="\t")

# Empty space in column names, oups
images.columns = [i.strip(" ") for i in images.columns]

# Parse unique concepts
concepts = []
for termlist in images.Terms:
  [concepts.append(t.strip(" ")) for t in termlist.split(",")]

concepts = numpy.unique(concepts).tolist()

# Generate label with task:contrast
labels = ["%s : %s" %(t[1].task,t[1].contrast) for t in images.iterrows()]
images.labels = labels

# Generate contrast by concept matrix
matrix = pandas.DataFrame(0,columns=concepts,index=images.labels)

# Make sure we don't have duplicates
if len(images["neurovault image"].unique()) != len(images["neurovault image"]):
    print "ERROR - duplicates!"

# We need to look up the file names with pyneurovault api
nv = api.NeuroVault()
urls = []
for image_id in images["neurovault image"]:
    urls.append(nv.images.file[nv.images.image_id == image_id].tolist()[0])

# Fill in the matrix!
for t in range(0,len(images.Terms)):
    label = images.labels[t]
    termlist = images.Terms[t]
    terms = [tt.strip(" ") for tt in termlist.split(",")]
    for term in terms:
        matrix.loc[label,term] = 1

matrix["files"] = urls
matrix["image_id"] = images["neurovault image"].tolist()
matrix.to_csv("data_matrix.tsv",sep="\t")
