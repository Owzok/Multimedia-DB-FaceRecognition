import face_recognition                      # main algorithm
import pickle                                # deserialize data
import cv2                                   # opencv, image processment
import faiss                                 # facebook ai search similarity search
import numpy as np                           # transform image data to img
from rtree import index                      # rtree indexing capanilities
from heapq import heappop, heapreplace, heappush, heapify # heap for knn

prop = index.Property()
prop.dimension = 128
prop.idx_extension = 'rtreeidx'
prop.filename = 'rindex'
prop.storage = index.RT_Disk

# TODO: rationalize the paths for rindex
# this does not have ../ because its run from app, which is in the parent directory
idx = index.Index('rindex',properties = prop)



# Simple Face recognition without FAISS or RTree
def detect_faces(img):
    data = pickle.loads(open("encodings.pickle", "rb").read())

    # Convert the image file to a NumPy array
    img_array = np.frombuffer(img, np.uint8)

    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            matched_indexes = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matched_indexes:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            
            name = max(counts, key=counts.get)
            #print("Found face: {}".format(name))
        names.append(name)

    f_str = "\n"
    for i in names:
        f_str += i + " "

    return f_str


def encode_image(img):
    '''
    Takes: an image
    Returns: its characteristic vector
    '''
    img_array = np.frombuffer(img, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)

    return encodings[0]


# Implementation of KNN with Facebook AI Similarity Search library
def faiss_knn(img, k):
    r_data = pickle.loads(open("encodings.pickle", "rb").read())

    data = np.array(r_data['encodings'])
    d = data.shape[1]

    index = faiss.IndexFlatL2(d)
    index.add(data)

    img_array = np.frombuffer(img, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)

    d = np.array([])

    for encoding in encodings:
        d = np.array([encoding])

    D, I = index.search(d, k) 

    images = []
    f_names = []

    m_index = 0
    for i in I[0]:
        images.append("../" + r_data['paths'][i])
        f_names.append(r_data['names'][i].replace("_", " "))


    # returns ([paths],[names],vector,path)
    return images[1:], f_names[1:], D[0], images[0]


def rindex_knn(img, k):
    #process the image into the vector

    encodings = encode_image(img)

    #print("facerec.rindex_knn [DEBUG] Preprocessed Image")
    #performa similarity search. Returns a generator. 
    #generator yields something so that (name,path,vector)

                                #expecta a bounding box
    nearest_people = idx.nearest(list(encodings)+list(encodings),k,'raw')

    #print('facerec.rindex_knn [DEBUG] Found Images')
    # adapt the returned data into standard return format

    '''
    Return format is as follows
    ([paths],        paths to the images
    [names],         names of the people recovered
    vector,          vector of similarity scores (inaplicable)
    path             for the 0th image.
    '''

    paths = []
    names = []
    path = None

    i = 0
    for image in list(nearest_people):
        if i == 0: #special case for nearest pic
            path = "../" + image[1]
        else:
            paths.append("../" + image[1])
            names.append(image[0].replace("_", " "))
        i=i+1

    #print('facerec.rindex_knn [DEBUG] Formatted response')
    # TODO: similarity scores on Rtree. Potentially non-implementable
    # in a sensible way
    vector = np.array([0]*len(paths))



    return (paths,names,vector,path)

def unindexed_knn(img, k):

    #get image encodings

    encoding = encode_image(img)

    #load data generator

    data = pickle.loads(open("encodings.pickle", "rb").read())

    k_nearest = [] # heap of (-distance,idx)
    heapify(k_nearest)

    #do the search

    i = 0
    for vec in data['encodings']:
        dist = np.linalg.norm(vec - encoding)

        if (len(k_nearest) < k+1):
            heappush(k_nearest,(-dist,i))
        elif (-k_nearest[0][0] > dist):
            heapreplace(k_nearest,(-dist,i))
        i=i+1
    
    # build the return values:

    results = [heappop(k_nearest) for i in range(len(k_nearest),0,-1)]

    images = []
    names  = []
    scores = []


    for i in results:
        images.append('../' + data['paths'][i[1]])
        names.append(data['names'][i[1]].replace("_", " "))
        scores.append(-i[0])
    
    images.reverse()
    names.reverse()
    scores.reverse()

    return images[1:], names[1:], np.array(scores), images[0]