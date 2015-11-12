## CS 2120 Assignment #4 -- Data Analysis
## Name: Aaron Leung
## Student number: 250724439

import numpy
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from pylab import *
import pdb
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import svm


def loaddata(filename="Real_Teens_Gaming_2008.csv"):
	"""
	This function loads the file `Real_Teens_Gaming_2008.csv` and returns a list of lists as subject_list
	:param: the csv file with all the data
	"""
	import csv
	
	reader = csv.reader(open(filename, 'rU'))
	subject_list = []
	
	for r in reader:
		subject_list.append(r)
		
	return subject_list
	
def readdata(subject_list):
	"""
	This function takes the lists of lists and makes it a numpyarray
	:param subject_list: the lists of lists with all the data from the csv file
	"""
	
	a = numpy.array(subject_list)
	#Take every row except for the first row with all the headers
	
	b = a[1:,:]
	#Makes all the values in array as a float
	real_numpy = b.astype(float)	
	
	return real_numpy
	
def visualize1(real_numpy,column1,column2,subject_list):
	"""
	Function takes two columsn and then compares the 
	:param real_numpy: numpy array of the data without the headers
	:param column1: first column of interest
	:param column2: second column of interest
	:param subject_list: numpy array of the data with headers:
	"""
	
	#initialize variables to count how many individuals who play certain games (dependent on the column number) also vote in an election
	list_for_column1 = []
	list_for_column2 = []
	
	percentage_values = [0.0,0.0]
	
	#The headers of the columns
	Kind_of_games = (subject_list[0][column1],subject_list[0][column2])
	y_pos = numpy.arange(len(Kind_of_games))
	
	#If the subject plays a certain genre of games AND voted in the last election, the counter list gets bigger accordingly by 1 (by adding 1.0 to the list)
	for subject in real_numpy:
		if subject[0] == 1 and subject[column1] == 1:
			list_for_column1.append(float(1.0))
		if subject[0] == 1 and subject[column2] == 1:
			list_for_column2.append(float(1.0))
	
	#Take the Percentages
	percentage_values[0] = (float(len(list_for_column1)))/(float(len(real_numpy)))*100.0
	percentage_values[1] = (float(len(list_for_column2)))/(float(len(real_numpy)))*100.0
	
	#Plots it on the graph
	performance = [percentage_values[0],percentage_values[1]]	
	
	
	plt.barh(y_pos, performance, align='center', alpha=0.4)
	plt.yticks(y_pos, Kind_of_games)
	plt.xlabel('Percentage(%)')
	plt.title('Percentage of people who play certain games and participate in an election')

	plt.show()
	
def visualize2(real_numpy):
	"""
	This function categorizes subjects based on the amount of time they spend playing video games and whether or not they voted in the last election
	:param real_numpy: the entire data without headers
	"""

	#Used to find the total number of subjects who play for a certain duration
	number_counter_for_1hour = 0
	number_counter_for_2hour = 0
	number_counter_for_3hours = 0
	number_counter_for_4hours = 0
	number_counter_for_5hours = 0
	number_counter_for_6hours = 0
	
	#Used to find the total number of subjects who did vote in the last election according to the amount of time they spend playing video games  
	election1hour = 0
	election2hour = 0
	election3hour = 0
	election4hour = 0
	election5hour = 0
	election6hour = 0
	
	percentage_values = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

	#Finds subjects who play for a certain duration and then finds out the subject who have voted in the last election
	for subject in real_numpy:
		if subject[14] == 1:
			number_counter_for_1hour += 1
			if subject[0] == 1:
				election1hour += 1
		if subject[14] ==2:
			number_counter_for_2hour += 1
			if subject[0] == 1:
				election2hour += 1
		if subject[14] == 3:
			number_counter_for_3hours += 1
			if subject[0] == 1:
				election3hour +=1
		if subject[14] == 4:
			number_counter_for_4hours += 1
			if subject[0] ==1:
				election4hour +=1
		if subject[14] ==5:
			number_counter_for_5hours +=1
			if subject[0] ==1:
				election5hour += 1
		if subject[14] ==6:
			number_counter_for_6hours +=1
			if subject[0] == 1:
				election6hour += 1
	
	#The y-value axis labels
	Number_of_hours_played = (1,2,3,4,5,6)
	y_pos = numpy.arange(len(Number_of_hours_played))
	
	# Finds the actual x-value of the percentage of subjects who have voted in the last election	
	percentage_values[0] = (float(election1hour))/(float(number_counter_for_1hour))*100.0
	percentage_values[1] = (float(election2hour))/(float(number_counter_for_2hour))*100.0
	percentage_values[2] = (float(election3hour))/(float(number_counter_for_3hours))*100.0
	percentage_values[3] = (float(election4hour))/(float(number_counter_for_4hours))*100.0
	percentage_values[4] = (float(election5hour))/(float(number_counter_for_5hours))*100.0
	percentage_values[5] = (float(election6hour))/(float(number_counter_for_6hours))*100.0
	
	performance = [percentage_values[0],percentage_values[1],percentage_values[2],percentage_values[3],percentage_values[4],percentage_values[5]]	
	
	
	plt.barh(y_pos, performance, align='center', alpha=0.4)
	plt.yticks(y_pos, Number_of_hours_played)
	plt.xlabel('Percentage(%)')
	plt.title('Percentage of teens who play video games for varying durations and participate in the election')

	plt.show()

def learn1(real_numpy):
	"""
	This function uses the k-nearest neighbour method to attempt to classify the data
	:param real_numpy: the entire data set without the headers
	"""
	
	#Splice so that data does not include the "labels" and that labels is just labels
	data = real_numpy[:,1:]
	labels = real_numpy[:,[1]]
	
	#Takes only the first 900 subjects to train on
	train_data = data[:900]
	train_labels = labels[:900]

	#Takes the rest of the subjects to test the method on 
	test_data = data[900:]
	test_labels = labels[900:]

	knn = KNeighborsClassifier()

	knn.fit(train_data, train_labels)
	
	count=0
	correct=0

	#If the prediction is the same as the label for that particular subject, then the correct counter goes up by 1
	for i in range(test_data.shape[0]):
		if knn.predict(test_data[i]) == test_labels[i]:
			correct += 1
		count +=1
	
	#Gets the success rate of the supervised k-nearest neighbors
	return float(float(correct)/float(count))
	
def learn2(real_numpy):
	"""
	This function uses the supervised Support Vector Machines method to try and accurately predict a label on a subject's characteristics(data)
	:param real_numpy: the entire data set without the headers
	"""

	#Splice so that data does not include the "labels" and that labels is just labels
	data = real_numpy[:,1:]
	labels = real_numpy[:,[1]]
	
	#Takes only the first 900 subjects to train on
	train_data = data[:900]
	train_labels = labels[:900]

	#Takes the rest of the subjects to test the method on 
	test_data = data[900:]
	test_labels = labels[900:]

	svc = svm.SVC(kernel = 'linear')
	svc.fit(train_data,train_labels)

	count=0
	correct=0

	#If the prediction is the same as the label for that particular subject, then the correct counter goes up by 1
	for i in range(test_data.shape[0]):
		if svc.predict(test_data[i]) == test_labels[i]:
			correct += 1
		count +=1
	
	#Gets the success rate of the supervised k-nearest neighbors
	return float(float(correct)/float(count))