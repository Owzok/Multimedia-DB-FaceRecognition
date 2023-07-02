# TODO
# Delete data
# Generate encodings
# Generate RTree
import pickle
from rtree import index


def generateRtreeFromEncodings(encodings_path: str, rtree_path: str):

    prop = index.Property()
    prop.dimension = 128
    prop.idx_extension = 'rtreeidx'
    prop.overwrite = True

    idx = index.Index(rtree_path,properties = prop)

    data = pickle.loads(open(encodings_path, "rb").read())

    for i in range(0, len(data['paths'])):
        point = list(data['encodings'][i]) + list(data['encodings'][i])

        idx.insert(i,point,obj=(data['names'][i],data['paths'][i])) #stores a tuple of a name, path and encoding

generateRtreeFromEncodings('../encodings.pickle','../rindex')