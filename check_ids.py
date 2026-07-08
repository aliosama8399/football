import re

def get_js_ids(path):
    with open(path, 'r') as f:
        js = f.read()
    ids = set(re.findall(r"getElementById\([\"']([^\"']+)[\"']\)", js))
    return ids

def get_html_ids(path):
    with open(path, 'r') as f:
        html = f.read()
    ids = set(re.findall(r'id="([^"]+)"', html))
    return ids

js_ids = get_js_ids('api/static/app.js')
html_ids = get_html_ids('api/static/index.html')

missing_in_html = js_ids - html_ids
unused_in_html = html_ids - js_ids

print(f"JS getElementById references: {len(js_ids)}")
print(f"HTML ID attributes: {len(html_ids)}")
print(f"\nIDs referenced in JS but MISSING from HTML:")
for i in sorted(missing_in_html):
    print(f"  MISSING: {i}")
print(f"\nHTML IDs never used in JS (informational):")
for i in sorted(unused_in_html):
    print(f"  unused: {i}")
