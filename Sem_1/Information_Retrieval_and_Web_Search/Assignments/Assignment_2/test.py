a = {'word': {1: 2}}

a['word'].update({2: 3})

for key in a['word']:
    print(key, a['word'][key])
