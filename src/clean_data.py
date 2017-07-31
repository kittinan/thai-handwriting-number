import os

directory = "../data/clean"

try:
    os.stat(directory)
except:
    os.mkdir(directory)

dirtys = [
    '7c9108fe-b240-4632-a024-f1ee922962ec',
    '20_a2178975-acff-4afe-88b9-f6fee8694ceb',
    'de366cab-6532-42ed-9926-38351927019b',
    '76c2e443-c8d1-40b0-96a9-073548c9617b',
    '21_e95ad3b8-30cb-47ad-9f68-2a5bb7aeb5bb',
    'e95ad3b8-30cb-47ad-9f68-2a5bb7aeb5bb',
    'fb05cb2a-c27b-4476-8cff-74f5ddbc8224',
    '078c1b18-e672-466d-a30b-f49a81710be6',
    '67ce79dc-de9c-4956-ad7b-fabf7aa9c6fa',
    '729207eb-f3f7-46e2-986a-74f990296da4',
    '420994cc-5e99-42eb-84b6-2392486a33b6',
]

os.system('cp -r ../data/raw/thai-handwriting-number.appspot.com/* ../data/clean/')
os.system('mv ../data/clean/10 ../data/clean/0')

for i in range(0 , 10):
    for dirty in dirtys:
        path = directory + '/' + str(i) + '/' + dirty + '.png'
        os.remove(path)