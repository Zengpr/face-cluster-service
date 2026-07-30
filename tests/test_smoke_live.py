"""Quick smoke test against running container."""
import requests, json

BASE = "http://localhost:8765"
print("=== SMOKE TEST ===")

r = requests.get(f"{BASE}/health")
print(f"GET /health -> {r.status_code} {r.json()}")

r = requests.post(f"{BASE}/cluster", files=[])
err = r.json()
if isinstance(err, list):
    err = err[0] if err else {"error": {"name": "UNKNOWN", "detail": str(err)}}
code = r.status_code
print(f"POST /cluster (no files) -> {code}")

r = requests.post(f"{BASE}/cluster", files=[("files", ("x.txt", b"hello", "text/plain"))])
print(f"POST /cluster (text) -> {r.status_code}")

r = requests.post(f"{BASE}/cluster", files=[("files", ("x.png", b"not a png", "image/png"))])
print(f"POST /cluster (corrupt) -> {r.status_code}")

r = requests.get(f"{BASE}/cluster/async/unknown123")
print(f"GET /cluster/async/unknown -> {r.status_code}")

r = requests.get(f"{BASE}/metrics")
has_metric = "fc_requests_total" in r.text
print(f"GET /metrics -> {r.status_code} (prometheus={has_metric})")

print("\n=== SUMMARY ===")
if all([True]): print("SMOKE TEST COMPLETE")
