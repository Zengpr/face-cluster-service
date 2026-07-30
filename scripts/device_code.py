import requests, urllib.parse, sys
r = requests.post('https://github.com/login/device/code',
                  data={'client_id': '178c6fc778ccc68e1d6a', 'scope': 'repo,workflow'})
params = dict(urllib.parse.parse_qsl(r.text))
print(f"Code: {params['user_code']}")
print(f"URL:  {params['verification_uri']}")
sys.stdout.flush()
