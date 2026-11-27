'''
what are cost functions?
Cost functions, also known as loss functions, are used to measure how well a machine learning model is performing. 
They quantify the difference between the predicted output and the actual target values.

Common examples of cost functions include:
- Mean Squared Error (MSE) for regression tasks
- Cross-Entropy Loss for classification tasks
- Hinge Loss for support vector machines

MSE : A classic Cost function
To prevent positive and negative value cancelling out and to penalize big misses, MSE squares the difference between the predicted and actual values.

steps:
  1. find each error : predicted - actual
  2. square each error (make everything +ve and punish large mistakes)
  3. Take the average across all examples
'''

actual_prices = [80.0, 50.0, 120.0]     # in ₹ lakhs
predicted_prices = [65.0, 52.0, 110.0]  # model guesses

squared_errors = [(p - a) ** 2 for p, a in zip(predicted_prices, actual_prices)]
mse_cost = sum(squared_errors) / len(squared_errors)

print("Squared errors:", squared_errors)  # [225.0, 4.0, 100.0]
print("Cost (MSE):    ", mse_cost)        # 109.67

'''
Cost function vs Loss function
A loss function measures the error for a single training example, while a cost function is the average of the loss over the entire training dataset.
For example, in MSE:
- Loss for one example: (predicted - actual) ** 2
- Cost for the dataset: mean of all individual losses

Differnt task requires different cost functions. For example:
- Regression tasks often use Mean Squared Error (MSE) or Mean Absolute Error (MAE) 
- Classification tasks often use Cross-Entropy Loss or Log Loss

'''

