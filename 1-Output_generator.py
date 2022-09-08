import numpy as np

dataDir='./data/'
file_name_test=dataDir + 'testTrack_hierarchy.txt'
file_name_train=dataDir + 'trainIdx2_matrix.txt'
output_file=dataDir + 'output22.txt'

fTest= open(file_name_test, 'r')
fTrain=open(file_name_train, 'r')
Trainline= fTrain.readline()
fOut = open(output_file, 'w')

trackID_vec=[0]*6
albumID_vec=[0]*6
artistID_vec=[0]*6
genreID_vec=[]
genreLens=[]

lastUserID=-1

last_shape = (6,25)
user_rating_inTrain=np.zeros(last_shape)

arr_li = []
for line in fTest:
    arr_test=line.strip().split('|')
    arr_li.append(len(arr_test))
    userID= arr_test[0]
    trackID= arr_test[1]
    albumID= arr_test[2]
    artistID=arr_test[3]
    genreID = [0]*21

    for genre in range(4, len(arr_test)):
        genreID.append(int(arr_test[genre]))

    if userID != lastUserID:
        genreID_vec = []
        genreLens = []
        ii=0


    trackID_vec[ii]=trackID
    albumID_vec[ii]=albumID
    artistID_vec[ii]=artistID
    genreID_vec += genreID
    genreLens += [len(genreID)]

    ii=ii+1
    lastUserID=userID

    if ii==6 : 
        user_rating_inTrain=np.zeros(shape=(6,max(genreLens)+2))
        while(Trainline):
            arr_train = Trainline.strip().split('|')
            trainUserID=arr_train[0]
            trainItemID=arr_train[1]
            trainRating=arr_train[2]
            Trainline=fTrain.readline()
            gens = genreID_vec.copy()

            if trainUserID < userID:
                continue
            if trainUserID == userID:
                for nn in range(0, 6):
                    if trainItemID==albumID_vec[nn]:
                        user_rating_inTrain[nn, 0]=trainRating
                    if trainItemID==artistID_vec[nn]:
                        user_rating_inTrain[nn, 1]=trainRating
                    for genre_num in range(genreLens[nn]):
                        if int(trainItemID) == gens.pop(0):
                            user_rating_inTrain[nn, genre_num+2]=trainRating

            if trainUserID> userID:
                for nn in range(0, 6):
                    outStr=str(userID) + '|' + str(trackID_vec[nn])+ '|' + str(user_rating_inTrain[nn,0]) + '|' + str(user_rating_inTrain[nn, 1])
                    for genre_num in range(genreLens[nn]):
                        outStr += '|' + str(user_rating_inTrain[nn,genre_num+2])
                    fOut.write(outStr + '\n')
                break

fTest.close()
fTrain.close()

