import requests

url = "http://127.0.0.1:5000/chat"
user_message = {"message": "I have a headache"}

response = requests.post(url, json=user_message)
print(response.json())  # AI assistant response
