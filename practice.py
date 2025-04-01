contacts = [
    ('James', 42),
    ('Amy', 24),
    ('John', 31),
    ('Amanda', 63),
    ('Bob', 18)
]

dict_contact = {}
for i in range(len(contacts)):
    dict_contact[contacts[i][0]] = contacts[i][1]

name = input()
age = dict_contact.get(name)
if age != None:
    print(f'{name} is {age}')
else:
    print('Not Found')