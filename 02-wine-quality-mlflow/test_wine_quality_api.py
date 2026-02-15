import requests, json

url = "http://127.0.0.1:5001/invocations"
headers = {"Content-Type": "application/json"}

testdata = {
    "dataframe_split": {
        "columns": [
            "fixed acidity","volatile acidity","citric acid","residual sugar",
            "chlorides","free sulfur dioxide","total sulfur dioxide",
            "density","pH","sulphates","alcohol"
        ],
        "data": [[7.0,0.27,0.36,20.7,0.045,45.0,170.0,1.001,3.0,0.45,8.8]]
    }
}

print("🍷 Testing Wine Quality Prediction API...")
response = requests.post(url, headers=headers, data=json.dumps(testdata))
print("Status:", response.status_code)
print("Body:", response.text)  # will show "Request processing time exceeded limit"

if response.ok:
    print("Prediction JSON:", response.json())
