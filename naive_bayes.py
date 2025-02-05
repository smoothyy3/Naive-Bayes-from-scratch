from math_functions import *

class Naive_Bayes:
    def __init__(self, laplace_smoothing = False):
        self.laplace = laplace_smoothing
        self.prior_probs = {}
        self.conditional_probs = {}
        self.numeric_params = {}

    # junge junge
    def fit(self, X_train, y_train):
        self.prior_probs = prior_prob(y_train)
    
        self.numeric_params = {}

        for feature_index in range(len(X_train[0])):
            if isinstance(X_train[0][feature_index], (int, float)):  # Only process numerical features (int/float)
                
                # values of the feature for each class
                feature_values_by_class = {}
                for i in range(len(y_train)):
                    label = y_train[i]
                    if label not in feature_values_by_class:
                        feature_values_by_class[label] = []
                    feature_values_by_class[label].append(X_train[i][feature_index])  # Append feature value
                
                # for each class, calculate mean and standard deviation
                self.numeric_params[feature_index] = {}
                for label, feature_values in feature_values_by_class.items():
                    mean_value = mean(feature_values)
                    std_value = std(feature_values)
                    self.numeric_params[feature_index][label] = (mean_value, std_value)
        
        # For categorical features, calculate the conditional probability of each feature value given the class label.
        self.conditional_probs = {}
        
        for feature_index in range(len(X_train[0])):
            # Only process categorical features (not int/float)
            if not isinstance(X_train[0][feature_index], (int, float)):
                
                # store conditional probabilities for each feature and class
                self.conditional_probs[feature_index] = {}
                
                # calculate conditional probabilities for each value of the feature
                for target_class in self.prior_probs:
                    self.conditional_probs[feature_index][target_class] = {}
                    unique_values = set([X_train[i][feature_index] for i in range(len(X_train))])  # all unique feature values for the current feature
                    
                    for value in unique_values:
                        if self.laplace:
                            # Use Laplace Smoothing
                            self.conditional_probs[feature_index][target_class][value] = conditional_prob_laplace(X_train, y_train, feature_index, value, target_class, K=len(unique_values))
                        else:
                            # no Laplace Smoothing
                            self.conditional_probs[feature_index][target_class][value] = conditional_prob(X_train, y_train, feature_index, value, target_class)

    def predict_prob(self, X_test):
        class_probs = {}

        for test_point in X_test:
            test_point_tuple = tuple(test_point)
            test_p_prob = {}

            for target_class in self.prior_probs:
                prob = self.prior_probs[target_class]

                for feature_index in range(len(test_point)):
                    feature_value = test_point[feature_index]

                    # for numeric data
                    if isinstance(feature_value, (int, float)):
                        mean, std = self.numeric_params[feature_index].get(target_class, (0, 1))
                        prob *= gaussian_PDF(feature_value, mean, std)
                    # for categorical data
                    else:
                        prob *= self.conditional_probs[feature_index][target_class].get(feature_value, 1e-6)

                test_p_prob[target_class] = prob

            class_probs[test_point_tuple] = test_p_prob
        return class_probs
        
    def predict(self, X_test):
        probs = {}

        for test_point in X_test:
            test_point_tuple = tuple(test_point)
            probs[test_point_tuple] = {}

            # get probs and classes for all testpoints
            class_probs= self.predict_prob([test_point])[test_point_tuple]

            # choose highest prob
            max_prob_class = max(class_probs, key= class_probs.get)

            probs[test_point_tuple] = max_prob_class
        return probs