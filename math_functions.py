''' 
CALCULATIONS NEEDED FOR BOTH DATA TYPES

- Prior probability calculation
- Normalization
- Estimating π (Pi) using the Leibniz series
- Computing Eulers number (e) via the Taylor series (requires factorial)
'''

def prior_prob(labels: list) -> dict:
    if len(labels) == 0:
        raise ValueError("Labels list cannot be empty")
    
    counts = {}
    for x in labels:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    total = len(labels)
    priors = {}
    for key in counts:
        priors[key] = counts[key] / total
    return priors

def factorial(x: int):
    if x < 0:
        raise ValueError("X cannot be negative")
    fact = 1
    for num in range(2, x + 1):
        fact *= num
    return fact

def estimate_pi(terms):
    result = 0.0
    for n in range(terms):
        result += (-1.0)**n/(2.0*n+1.0)
    return 4*result

def taylor_e(terms):
    result = 0.0
    for n in range(terms):
        result += 1 / factorial(n)
    return result

E_APPROX = taylor_e(20)

''' 
NUMERIC FEATURE CALCULATIONS

- Mean calculation
- Variance calculation
- Standard deviation
- Gaussian probability density function (PDF)
'''

def mean(lst) -> float:
    if not lst:
        raise ValueError("List cannot be empty")
    
    result = 0
    for x in lst:
        result += x
    return result / len(lst)

def var(lst) -> float:
    if not lst:
        raise ValueError("List cannot be empty")
    
    lst_mean = mean(lst)
    result = 0
    for x in lst:
        result += (x - lst_mean)**2
    return result / len(lst)

def std(lst) -> float:
    if not lst:
        raise ValueError("List cannot be empty")
    
    return (var(lst))**0.5

def gaussian_PDF(x: float, mean_train_x: float, std_train_x: float) -> float:
    exp = -(((x - mean_train_x)**2) / (2 * std_train_x ** 2))
    norm_factor = 1 / ((2 * estimate_pi(10000) * (std_train_x ** 2)) ** 0.5)

    return norm_factor * (E_APPROX ** exp)

''' 
CATEGORICAL FEATURE CALCULATIONS

- Conditional probability
- laplace smoothing
'''

def conditional_prob(X_train, y_train, feature_index, value, target_class) -> float:
    # count how often the target class is in the class labels
    target_instances = 0
    for x in y_train:
        if x == target_class:
            target_instances += 1

    # count how often the feature value appears in the target class
    feature_instances = 0
    for i in range(len(y_train)):
        if y_train[i] == target_class and X_train[i][feature_index] == value:
            feature_instances += 1
    
    # avoid division by 0
    if target_instances == 0:
        return 0.0
    
    conditional_prob = (feature_instances) / (target_instances)
    return conditional_prob

def conditional_prob_laplace(X_train, y_train, feature_index, value, target_class, K) -> float:
    # count how often the target class is in the class labels
    target_instances = 0
    for x in y_train:
        if x == target_class:
            target_instances += 1

    # count how often the feature value appears in the target class
    feature_instances = 0
    for i in range(len(y_train)):
        if y_train[i] == target_class and X_train[i][feature_index] == value:
            feature_instances += 1
    
    # avoid division by 0
    if target_instances == 0:
        return 0.0
    
    conditional_prob = (feature_instances + 1) / (target_instances + K)
    return conditional_prob