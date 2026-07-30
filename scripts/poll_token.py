"""Poll device flow for access token."""
import requests, urllib.parse, time, sys

# Get a fresh device code
r = requests.post('https://github.com/login/device/code',
    data={'client_id': '178c6fc778ccc68e1d6a', 'scope': 'repo,workflow'})
params = dict(urllib.parse.parse_qsl(r.text))
dc = params['device_code']
interval = int(params.get('interval', 5))
print(f"Enter this code in browser: {params['user_code']}")
print(f"URL: {params['verification_uri']}")

for i in range(60):
    r = requests.post('https://github.com/login/oauth/access_token',
        data={
            'client_id': '178c6fc778ccc68e1d6a',
            'device_code': dc,
            'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
        })
    resp = dict(urllib.parse.parse_qsl(r.text))
    if 'access_token' in resp:
        print(f"NEW_TOKEN={resp['access_token']}")
        print(f"SCOPE={resp.get('scope','')}")
        sys.exit(0)
    err = resp.get('error', 'unknown')
    if err != 'authorization_pending' and err != 'slow_down':
        print(f"ERROR: {resp}")
        sys.exit(1)
    time.sleep(interval if err != 'slow_down' else interval + 5)

print("TIMEOUT")
sys.exit(1)
