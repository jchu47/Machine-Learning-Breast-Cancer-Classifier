import seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()
#Uncomment these lines to get an understanding of the data set
#print(breast_cancer_data.data[0])
#print(breast_cancer_data.feature_names)
#print(breast_cancer_data.target)
#print(breast_cancer_data.target_names)


#Splitting the data into training and testing sets
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)

classifier = KNeighborsClassifier (n_neighbors=9)

#Training the classifier
classifier.fit(training_data, training_labels)

print(classifier.score(validation_data, validation_labels))

k_list = range(1, 101)

#Looping throught possible K values to find the optimal value
accuracies = []
for k in range(1, 101):
  classifier = KNeighborsClassifier (n_neighbors=k)

  classifier.fit(training_data, training_labels)

  accuracies.append(classifier.score(validation_data, validation_labels))

#Plotting the data to visualize the optimal value of K
plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()



