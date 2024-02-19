# Based on similar historical data, the ML team has managed to create a model 
# that can accurately predict the price for the house given the parameters you have access to.

# They have provided you code for the inference process. 
# For each row you have saved in the CSV, determine the price of the house.

import torch
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


# Define the model architecture (must match the training phase)
class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.input_to_hidden = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layer_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_output = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_to_hidden(x))
        x = torch.relu(self.hidden_layer_1(x))
        x = torch.relu(self.hidden_layer_2(x))
        x = self.hidden_to_output(x)
        return x


# Load the saved model
def load_model(model_path, input_dim, hidden_dim, output_dim):
    model = LinearRegressionModel(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to inference mode
    return model


# Process a single row (utility function)
def process_row(row, X_scaler, model, y_scaler):
    # Convert row to numpy array and reshape for scaler
    row_array = np.array(row).astype(np.float32).reshape(1, -1)
    row_scaled = X_scaler.transform(row_array)
    row_tensor = torch.from_numpy(row_scaled)
    with torch.no_grad():  # No need to calculate gradients
        prediction = model(row_tensor)
    prediction_np = prediction.numpy()
    prediction_inverse_scaled = y_scaler.inverse_transform(prediction_np.reshape(-1, 1))
    # Return the first element of the prediction as a float
    return prediction_inverse_scaled[0]


if __name__ == "__main__":
    model_path = './model/real_estate_model.pth'  # Update path accordingly
    csv_path_for_inference = './data/real_estate_inference_examples.csv'  # Update path accordingly
    input_dim = 4  # Number of features
    hidden_dim = 50
    output_dim = 1

    # Load model
    model = load_model(model_path, input_dim, hidden_dim, output_dim)

    # Placeholder for X_scaler and y_scaler - ideally loaded from saved state
    X_scaler = joblib.load('./model/x_scaler.pkl')  # Assuming the scaler is pre-fitted and saved
    y_scaler = joblib.load('./model/y_scaler.pkl')

    

    # PART 1 your code goes here




# PART 2 
# Your product manager asks you for the following:

# She would like to be alerted if the mean price of the latest four processed houses is greater than a given threshold.
# She would like to be able to set the threshold dynamically. 
# The first iteration of the feature is to print "Condition met" if you have a positive match for the condition.

# PS - modify the code above.