import requests
import random

# API endpoint
API_URL = "https://asteroid-diameter.onrender.com/predict"

def generate_dummy_data():
    # Generate random values for each field
    data = {
        'a': random.uniform(1.0, 3.0),
        'e': random.uniform(0.0, 1.0),
        'i': random.uniform(0.0, 180.0),
        'om': random.uniform(0.0, 360.0),
        'w': random.uniform(0.0, 360.0),
        'q': random.uniform(0.5, 1.5),
        'ad': random.uniform(1.0, 3.0),
        'per_y': random.uniform(1.0, 10.0),
        'data_arc': random.uniform(1.0, 1000.0),
        'n_obs_used': random.randint(1, 1000),
        'H': random.uniform(10.0, 30.0),
        'albedo': random.uniform(0.0, 1.0),
        'moid': random.uniform(0.0, 1.0),
        'n': random.uniform(0.0, 1.0),
        'ma': random.uniform(0.0, 360.0),
        'approx_diameter': random.uniform(0.1, 10.0),
        'condition_code': random.randint(0, 9),
        'neo': random.choice(['Y', 'N']),
        'pha': random.choice(['Y', 'N']),
        'class': random.choice(['AMO', 'APO', 'AST', 'ATE', 'CEN', 'IMB', 'MBA', 'MCA', 'OMB', 'TJN', 'TNO'])
    }
    return data

def send_dummy_data():
    # Generate dummy data
    dummy_data = generate_dummy_data()
    
    # Send POST request to the API
    try:
        response = requests.post(API_URL, json=dummy_data)
        if response.status_code == 200:
            print("Success! Response:", response.json())
        else:
            print(f"Error {response.status_code}:", response.json())
    except requests.exceptions.RequestException as e:
        print("Request failed:", str(e))

if __name__ == '__main__':
    # Send 10 dummy requests
    for _ in range(10):
        send_dummy_data()