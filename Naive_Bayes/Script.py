import bayes

PostingList, ClassVector = bayes.loadDataSet()
print '*** PostingList ***'
print PostingList
print '*** ClassVector ***'
print ClassVector
myVocabList = bayes.createVocabList(PostingList)
print '*** myVocabList ***'
print myVocabList
trainMat = []
for postinDoc in PostingList:
	trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))

print '*** trainMatrix ***'
print trainMat
p0V, p1V, pAb = bayes.trainNB0(array(trainMat), array(ClassVector)

print '*** (c0/w) ***'
print p0V
print '*** (c1/w) ***'
print p1V

testEntry = ['love', 'my', 'dalmation']
thisDoc = array(bayes.setOfWords2Vec(myVocabList, testEntry))
if (classifyNB(thisDoc,p0V,p1V,pAb) == 0):
	print testEntry, 'classified as not Abusive :)'
else:
	print testEntry, 'classfied as Abusive :('

#print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb);
	
testEntry = ['stupid', 'garbage']
thisDoc = array(bayes.setOfWords2Vec(myVocabList, testEntry))
if (classifyNB(thisDoc,p0V,p1V,pAb) == 0):
	print testEntry, 'classified as not Abusive :)'
else:
	print testEntry, 'classfied as Abusive :('
#print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb);