import requests
BASE_URL = 'http://127.0.0.1:5000'

response = requests.get(f'{BASE_URL}/get_ip')
ip = response.json()['ip']
print(f'IP Address: {ip}')

response = requests.get(f'{BASE_URL}/get_iris')
f1 = response.json()['f1']
accuracy = response.json()['accuracy']
precision = response.json()['precision']
recall = response.json()['recall']
confusion_matrix = response.json()['confusion_matrix']
print('Iris Model Performance:')
print(f'f1-score: {f1}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print('Confusion Matrix: ')
print(confusion_matrix)

response = requests.get(f'{BASE_URL}/get_MNIST')
f1 = response.json()['f1']
accuracy = response.json()['accuracy']
precision = response.json()['precision']
recall = response.json()['recall']
confusion_matrix = response.json()['confusion_matrix']
print('Iris Model Performance:')
print(f'f1-score: {f1}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print('Confusion Matrix: ')
print(confusion_matrix)