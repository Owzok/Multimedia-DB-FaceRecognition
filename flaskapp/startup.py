# TODO
# Delete data
# Generate encodings
# Generate RTree
import pickle
import os
from rtree import index
import generate_encodings

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

def clean():
    try:
        os.remove('../rindex.dat')
        os.remove('../rindex.rtreeidx')
    except:
        pass

# this function truncates an exixting encoding file, thus reducing the ammount of stuff 
# without any recalculation of the VERY costly encodings
def truncateEncodings(oldFile,newFile, max_ammt):
    data = pickle.loads(open(oldFile, "rb").read())

    newdata = {"paths":data["paths"][0:max_ammt],
               "encodings":data["encodings"][0:max_ammt],
               "names":data["names"][0:max_ammt]}

    f = open(newFile,'wb')
    f.write(pickle.dumps(newdata))
    f.close()


clean()
#generate_encodings.generate_encodings()
truncateEncodings('./full_encodings.pickle','../encodings.pickle',12800)
generateRtreeFromEncodings('../encodings.pickle','../rindex')