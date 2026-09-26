import numpy as np

# Estimate the price of a 1650 sq-ft, 3 br house
def predict_multi(house_features, mu, sigma, theta):

    #   ====================== YOUR CODE HERE ======================
    # You should use the learned model with its characteristics (mu, sigma, theta)
    # to predict the price of a house based on its features (surface and bedrooms)

    house_features_norm = (house_features-mu)/sigma
    house_features_norm = np.concatenate((np.ones((1, 1)), house_features_norm), axis=1)
    price = house_features_norm.dot(theta)
    print(f'shapes {np.shape(theta)}  {np.shape(house_features_norm)}')
    return price
