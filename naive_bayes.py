from math_functions import *

class Naive_Bayes:
    def __init__(self, laplace_smoothing = False):
        self.laplace = laplace_smoothing
        self.prior_probs = {}
        self.conditional_probs = {}
        self.numeric_params = {}

    def fit(self, X_train, y_train):
        self.prior_probs = prior_prob(y_train)
    
        self.numeric_params = {}
        
        # Loop through each feature in X_train
        for feature_index in range(len(X_train[0])):  # Go through each feature
            if isinstance(X_train[0][feature_index], (int, float)):  # Only process numerical features (int/float)
                
                # Store values of the feature for each class in a dictionary
                feature_values_by_class = {}  # Dictionary to store feature values for each class
                for i in range(len(y_train)):  # Loop through all the training samples
                    label = y_train[i]  # Get the class label for this sample
                    if label not in feature_values_by_class:
                        feature_values_by_class[label] = []  # Initialize list for that class
                    feature_values_by_class[label].append(X_train[i][feature_index])  # Append feature value
                
                # for each class, calculate mean and standard deviation
                self.numeric_params[feature_index] = {}  # Store these parameters for each feature
                for label, feature_values in feature_values_by_class.items():
                    mean_value = mean(feature_values)  # Calculate mean for the feature in this class
                    std_value = std(feature_values)  # Calculate standard deviation for the feature in this class
                    self.numeric_params[feature_index][label] = (mean_value, std_value)  # Store the mean and std for the class
        
        # For categorical features, calculate the conditional probability of each feature value given the class label.
        self.conditional_probs = {}
        
        # Loop through each feature in X_train
        for feature_index in range(len(X_train[0])):  # Go through each feature
            if not isinstance(X_train[0][feature_index], (int, float)):  # Only process categorical features (not int/float)
                
                # Initialize a dictionary to store conditional probabilities for each feature and class
                self.conditional_probs[feature_index] = {}
                
                # Loop over each class to calculate conditional probabilities for each value of the feature
            for target_class in self.prior_probs:
                self.conditional_probs[feature_index][target_class] = {}
                unique_values = set([X_train[i][feature_index] for i in range(len(X_train))])  # Get all unique feature values for the current feature
                
                for value in unique_values:  # Loop over each unique value of the feature
                    if self.laplace:
                        # Use Laplace Smoothing
                        self.conditional_probs[feature_index][target_class][value] = conditional_prob_laplace(X_train, y_train, feature_index, value, target_class, K=len(unique_values))
                    else:
                        # no Laplace Smoothing
                        self.conditional_probs[feature_index][target_class][value] = conditional_prob(X_train, y_train, feature_index, value, target_class)

    #def predict_prob(self, X_test):
        
    #def predict(self, X_test):