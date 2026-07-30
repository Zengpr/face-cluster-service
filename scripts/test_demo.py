"""Generate test images and cluster via API inside container."""
import os, json
import cv2
import numpy as np
import requests
import tempfile

outdir = tempfile.mkdtemp()
for ident in range(3):
    for shot in range(3):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 200
        cv2.putText(img, f'ID{ident}_S{shot}', (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        cv2.imwrite(os.path.join(outdir, f'ident{ident}_shot{shot}.jpg'), img)

files = []
for p in sorted(os.listdir(outdir)):
    path = os.path.join(outdir, p)
    files.append(('files', (p, open(path, 'rb'), 'image/jpeg')))

try:
    r = requests.post('http://localhost:8000/cluster',
                      files=files,
                      headers={'X-Demo-Mode': 'true'},
                      timeout=120)
    result = r.json()
    print('Status:', r.status_code)
    if r.status_code == 200:
        print('n_clusters:', result.get('n_clusters'))
        print('n_images:', result.get('n_images'))
        print('clusters:', json.dumps(result.get('clusters', {}), indent=2))
        print('silhouette:', result.get('silhouette'))
        print('SUCCESS: clustering works end-to-end')
    else:
        print('Error:', json.dumps(result, indent=2)[:800])
finally:
    for _, fobj in files:
        fobj[1].close()
    for p in os.listdir(outdir):
        os.remove(os.path.join(outdir, p))
    os.rmdir(outdir)
