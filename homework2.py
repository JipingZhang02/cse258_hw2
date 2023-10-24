# %%
import random
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
from collections import defaultdict
import gzip
import dateutil.parser

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
answers = {}

# %%
def parseData(fname):
    for l in open(fname):
        yield eval(l)

# %%
data = list(parseData("beer_50000.json"))

# %%
random.seed(0)
random.shuffle(data)

# %%
dataTrain = data[:25000]
dataValid = data[25000:37500]
dataTest = data[37500:]

# %%
yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
yValid = [d['beer/ABV'] > 7 for d in dataValid]
yTest = [d['beer/ABV'] > 7 for d in dataTest]

# %%
categoryCounts = defaultdict(int)
for d in data:
    categoryCounts[d['beer/style']] += 1

# %%
categories = [c for c in categoryCounts if categoryCounts[c] > 1000]

# %%
catID = dict(zip(list(categories),range(len(categories))))

# %%
review_column_names = [kname for kname in data[0] if (("review" in kname) and isinstance(data[0][kname],float))]

# %%
max_text_len = max(len(datum['review/text']) for datum in data)

# %%
def feat(datum, includeCat = True, includeReview = True, includeLength = True, ret_np_array = False):
    res = list()
    if includeCat:
        catID_1hot_vector = [0.0]*(len(catID))
        style_this = datum['beer/style']
        if style_this in catID:
            catID_1hot_vector[catID[style_this]]=1.0
        res.extend(catID_1hot_vector)
    if includeReview:
        for review_col_name in review_column_names:
            res.append(datum[review_col_name]/5.0)
    if includeLength:
        res.append(len(datum['review/text'])/max_text_len)
    assert len(res)>0,f"the feat function returns no feature for datum {datum}"
    if ret_np_array:
        res = np.array(res,dtype=float)
    return res


# %%
# lin_reg = linear_model.LinearRegression()
# lin_reg.fit()
# lin_reg.predict()

def get_performance_info(y_actual,y_predict):
    if not isinstance(y_actual,np.ndarray):
        y_actual = np.array(y_actual)
    y_actual = y_actual.reshape((-1,))
    y_predict = y_predict.reshape((-1,))
    TP = np.sum((y_actual == 1) & (y_predict == 1))
    FP = np.sum((y_actual == 0) & (y_predict == 1))
    TN = np.sum((y_actual == 0) & (y_predict == 0))
    FN = np.sum((y_actual == 1) & (y_predict == 0))
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    FNR = FN / (TP + FN)
    BER = 1 - (0.5 * (TPR + TNR))
    return TP,FP,TN,FN,TPR, FPR, TNR, FNR, BER

def pipeline(reg, includeCat = True, includeReview = True, includeLength = True):
    get_x_row = lambda datum:feat(datum,includeCat=includeCat,includeReview=includeReview,includeLength=includeLength)
    get_all_x = lambda data:np.array(list(get_x_row(datum) for datum in data),dtype=float)
    x_train = get_all_x(dataTrain)
    x_valid = get_all_x(dataValid)
    x_test = get_all_x(dataTest)
    logisticRegModel = linear_model.LogisticRegression(class_weight="balanced",penalty="l2",C=reg)
    logisticRegModel.fit(x_train,yTrain)
    y_pred_valid = logisticRegModel.predict(x_valid)>=0.5
    y_pred_test = logisticRegModel.predict(x_test)>=0.5
    return logisticRegModel,get_performance_info(yValid,y_pred_valid)[-1],get_performance_info(yTest,y_pred_test)[-1]




# %%
mod, validBER, testBER = pipeline(10, True, False, False)

# %%
answers['Q1'] = [validBER, testBER]

# %%
mod2, validBER2, testBER2 = pipeline(10, True, True, True)
answers['Q2'] = [validBER2, testBER2]

# %%
best_c,best_model,min_ber_valid,ber_test = 0,None,1.0,1.0
for c in [0.001, 0.01, 0.1, 1, 10]:
    model,ber_valid,b_t_this = pipeline(c,True,True,True)
    if ber_valid<min_ber_valid:
        best_c = c
        best_model = model
        min_ber_valid = ber_valid
        ber_test = b_t_this

answers['Q3'] = [best_c,min_ber_valid,ber_test] 

# %%
mod, validBER, testBER_noCat = pipeline(1.0,False,True,True)
mod, validBER, testBER_noReview = pipeline(1.0,True,False,True)
mod, validBER, testBER_noLength = pipeline(1.0,True,True,False)
answers['Q4'] = [testBER_noCat, testBER_noReview, testBER_noLength]

# %%
path = "amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz"
f = gzip.open(path, 'rt', encoding="utf8")

header = f.readline()
header = header.strip().split('\t')

# %%
dataset = []

pairsSeen = set()

for line in f:
    fields = line.strip().split('\t')
    d = dict(zip(header, fields))
    ui = (d['customer_id'], d['product_id'])
    if ui in pairsSeen:
        # print("Skipping duplicate user/item:", ui)
        continue
    pairsSeen.add(ui)
    d['star_rating'] = int(d['star_rating'])
    d['helpful_votes'] = int(d['helpful_votes'])
    d['total_votes'] = int(d['total_votes'])
    dataset.append(d)

# %%
dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]

# %%
usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {} 
ratingDict = {} # (u,i)->r To retrieve a rating for a specific user/item pair
timeDict = {} # (u,i)->t
reviewsPerUser = defaultdict(list) # TODO: what is this?
# testSetDatas = set()

for d in dataTrain:
    try:
        item_id = d['product_id']
        user_id = d['customer_id']
        rating = d['star_rating']
        item_name = d['product_title']
        review_time = d['review_date']
        usersPerItem[item_id].add(user_id)
        itemsPerUser[user_id].add(item_id)
        itemNames[item_id] = item_name
        ratingDict[(user_id,item_id)]=rating
        timeDict[(user_id,item_id)]=dateutil.parser.parse(review_time).timestamp()
    except BaseException as e:
        print(f"error happened when dealing with {d} : str({e})")
        break



# %%
testSetDatas = set()
testSetRatingDict = {}
testSetTimeDict = {}

for d in dataTest:
    try:
        item_id = d['product_id']
        user_id = d['customer_id']
        rating = d['star_rating']
        item_name = d['product_title']
        review_time = d['review_date']
        testSetDatas.add((user_id,item_id))
        itemNames[item_id] = item_name
        testSetRatingDict[(user_id,item_id)]=rating
        testSetTimeDict[(user_id,item_id)]=dateutil.parser.parse(review_time).timestamp()
    except BaseException as e:
        print(f"error happened when dealing with {d} : str({e})")
        break


# %%
userAverages = {} # avg rating every user gives items he/she bought
itemAverages = {} # avg rating of every item given by users who bought it

for u in itemsPerUser:
    total_score,item_cnt=0,0
    for item_this_user_bought in itemsPerUser[u]:
        # if (u,item_this_user_bought) in testSetDatas:
        #     continue
        total_score+=ratingDict[(u,item_this_user_bought)]
        item_cnt+=1
    if item_cnt==0:
        continue
    userAverages[u] = total_score/item_cnt
    
for i in usersPerItem:
    total_score,user_cnt=0,0
    for user_bought_this_item in usersPerItem[i]:
        # if (user_bought_this_item,i) in testSetDatas:
        #     continue
        total_score+=ratingDict[(user_bought_this_item,i)]
        user_cnt+=1
    if user_cnt==0:
        continue
    itemAverages[i] = total_score/user_cnt

ratingMean = sum(r for _k,r in ratingDict.items())/len(ratingDict)

# %%
# def Jaccard(item1, item2):
#     u_i1_set = usersPerItem[item1]
#     u_i2_set = usersPerItem[item2]
#     return len(u_i1_set.intersection(u_i2_set))/len(u_i1_set.union(u_i2_set))

def Jaccard(s1, s2):
    if len(s1)+len(s2)==0:
        return 0
    return len(s1.intersection(s2))/len(s1.union(s2))

# %%
def mostSimilar(i, N):
    simWithItemId = []
    u_i_set = usersPerItem[i]
    for j,u_j_set in usersPerItem.items():
        if j==i:
            continue
        sim_this = Jaccard(u_i_set,u_j_set)
        simWithItemId.append((sim_this,j))
    simWithItemId.sort(key=lambda tup:tup[0],reverse=True)
    return simWithItemId[:N]

# %%
query = 'B00KCHRKD6'
ms = mostSimilar(query, 10)
answers['Q5'] = ms

# %%
# print(itemNames[query])
# for _sim,item_id in ms:
#     print(itemNames[item_id])

# %%
def MSE(y, ypred):
    if not isinstance(y,np.ndarray):
        y = np.array(y)
    if not isinstance(ypred,np.ndarray):
        ypred = np.array(ypred)
    return np.sum((y-ypred)**2)/len(y)

# %%
def predictRating(user,item):
    uISet = usersPerItem[item]
    if item not in itemAverages:
        return ratingMean
    itemAvgRating = itemAverages[item]
    deltaRatingWeightedSum,weightSum=0.0,0.0
    for j in itemsPerUser[user]:
        if j==item:
            continue
        similarity = Jaccard(uISet,usersPerItem[j])
        deltaRatingWeightedSum+=similarity*(ratingDict[(user,j)]-itemAverages[j])
        weightSum+=similarity
    if weightSum==0.0:
        return itemAvgRating
    return itemAvgRating+deltaRatingWeightedSum/weightSum


# %%
simPredictions = list(predictRating(u,i) for (u,i) in testSetDatas)
labels = list(testSetRatingDict[ui_tuple] for ui_tuple in testSetDatas)

# %%
answers['Q6'] = MSE(simPredictions, labels)

# %%
all_times_in_train_set = list(map(lambda tup:tup[1],timeDict.items()))

# %%
RAND_TIMES = 1000
total_time_span = 0.0
for _ in range(RAND_TIMES):
    i = random.randint(0,len(all_times_in_train_set))
    j = random.randint(0,len(all_times_in_train_set))
    total_time_span += abs(all_times_in_train_set[i]-all_times_in_train_set[j])

# %%
avg_time_span = total_time_span/RAND_TIMES

# %%
def predictRatingWithTimeFactor(user,item,timeFactorFunc):
    uISet = usersPerItem[item]
    if item not in itemAverages:
        return ratingMean
    itemAvgRating = itemAverages[item]
    deltaRatingWeightedSum,weightSum=0.0,0.0
    for j in itemsPerUser[user]:
        if j==item:
            continue
        similarity = Jaccard(uISet,usersPerItem[j])
        weight = timeFactorFunc(user,item,j) * similarity
        deltaRatingWeightedSum+=weight*(ratingDict[(user,j)]-itemAverages[j])
        weightSum+=weight
    if weightSum==0.0:
        return itemAvgRating
    return itemAvgRating+deltaRatingWeightedSum/weightSum

USE_TQDM = True

krange = range(-20,21)
if USE_TQDM:
    import tqdm
    krange = tqdm.tqdm(krange)

lowest_mse,best_lambda,best_k = 2000000000,None,None
e=2.718281828
for k in krange:
    decay_lambda_inverse = (2**k)*avg_time_span
    timeFactorFunc = lambda u,i,j:e**(-abs(timeDict[(u,j)]-testSetTimeDict[(u,i)])/decay_lambda_inverse)
    simPredictions = list(predictRatingWithTimeFactor(u,i,timeFactorFunc) for (u,i) in testSetDatas)
    labels = list(testSetRatingDict[ui_tuple] for ui_tuple in testSetDatas)
    mse_this = MSE(simPredictions,labels)
    if mse_this<lowest_mse:
        lowest_mse = mse_this
        best_k = k
        best_lambda = 1/decay_lambda_inverse
    

# %%
lowest_mse

# %%
best_k

# %%
answers['Q7'] = ["In this problem, I used the teacher's suggested timeFactorFunction, f(t)=exp(-lambda*t).\n"+
                 "To make it reasonable, lambda inverse should be comparable to time span between purchases\n"+
                 "So I first used random sampling to estimate the average time_span between purchases\n"+
                 "Then searched lambda inverse through [2^(-20)~2^(20)]*avg_time_span and see which has best effect\n"+
                 "Finally I found 2^(-14)*avg_time_span as lambda inverse is the best\n"+
                 "Which means we should give recent purchase a much bigger weight than earlier purchases.\n"
                 , lowest_mse]

# %%
f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()


