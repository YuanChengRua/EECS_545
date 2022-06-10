import json


def read_json(json_file_name):
    with open(json_file_name, 'r') as j:
        contents = json.loads(j.read())
    return contents


print(read_json('json_tree.json'))
