import json

# Replace this with the contents of your original credentials file
with open("credentials_austin.json", "r") as f:
    data = f.read()

escaped = data.replace("\n", "\\n").replace('"', '\\"')
escaped_json = json.loads(data)
escaped_json['private_key'] = escaped_json['private_key'].replace("\n", "\\n")

print(json.dumps(escaped_json))
