"""Generate a Postman collection matching the API surface + test cases"""
from __future__ import annotations

import json
from pathlib import Path


COLLECTION = {
    "info": {
        "name": "Face Cluster Service",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
    },
    "item": [
        {
            "name": "Health",
            "request": {
                "method": "GET",
                "url": {"raw": "{{BASE}}/health", "host": ["{{BASE}}"], "path": ["health"]},
            },
        },
        {
            "name": "Ready",
            "request": {
                "method": "GET",
                "url": {"raw": "{{BASE}}/ready", "host": ["{{BASE}}"], "path": ["ready"]},
            },
        },
        {
            "name": "Metrics",
            "request": {
                "method": "GET",
                "url": {"raw": "{{BASE}}/metrics", "host": ["{{BASE}}"], "path": ["metrics"]},
            },
        },
        {
            "name": "Cluster happy path (2 face images)",
            "request": {
                "method": "POST",
                "url": {"raw": "{{BASE}}/cluster", "host": ["{{BASE}}"], "path": ["cluster"]},
                "body": {
                    "mode": "formdata",
                    "formdata": [
                        {"key": "files", "type": "file",
                         "src": ["tests/data/images/ident0_shot0.png"]},
                        {"key": "files", "type": "file",
                         "src": ["tests/data/images/ident0_shot1.png"]},
                        {"key": "threshold", "type": "text", "value": "0.6"},
                    ],
                },
            },
        },
        {
            "name": "Cluster no files (expect 400 NO_IMAGES)",
            "request": {
                "method": "POST",
                "url": {"raw": "{{BASE}}/cluster", "host": ["{{BASE}}"], "path": ["cluster"]},
            },
        },
        {
            "name": "Cluster invalid threshold (expect 400 INVALID_THRESHOLD)",
            "request": {
                "method": "POST",
                "url": {"raw": "{{BASE}}/cluster", "host": ["{{BASE}}"], "path": ["cluster"]},
                "body": {
                    "mode": "formdata",
                    "formdata": [
                        {"key": "files", "type": "file",
                         "src": ["tests/data/images/ident0_shot0.png"]},
                        {"key": "threshold", "type": "text", "value": "9.9"},
                    ],
                },
            },
        },
        {
            "name": "Cluster async submit + poll",
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "type": "text/javascript",
                        "exec": [
                            "const json = pm.response.json();",
                            "pm.collectionVariables.set('TASK_ID', json.task_id);",
                            "console.log('TASK_ID=' + json.task_id);",
                        ],
                    },
                }
            ],
            "request": {
                "method": "POST",
                "url": {"raw": "{{BASE}}/cluster/async",
                        "host": ["{{BASE}}"], "path": ["cluster", "async"]},
                "body": {
                    "mode": "formdata",
                    "formdata": [
                        {"key": "files", "type": "file",
                         "src": ["tests/data/images/ident0_shot0.png"]},
                        {"key": "files", "type": "file",
                         "src": ["tests/data/images/ident0_shot1.png"]},
                    ],
                },
            },
        },
        {
            "name": "Cluster async status (poll)",
            "request": {
                "method": "GET",
                "url": {"raw": "{{BASE}}/cluster/async/{{TASK_ID}}",
                        "host": ["{{BASE}}"], "path": ["cluster", "async", "{{TASK_ID}}"]},
            },
        },
        {
            "name": "Cluster unknown task (expect 404 TASK_NOT_FOUND)",
            "request": {
                "method": "GET",
                "url": {"raw": "{{BASE}}/cluster/async/unknown-xyz",
                        "host": ["{{BASE}}"], "path": ["cluster", "async", "unknown-xyz"]},
            },
        },
    ],
    "variable": [{"key": "BASE", "value": "http://localhost:8000"}],
}


if __name__ == "__main__":
    out = Path("tests/postman/face-cluster.postman_collection.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(COLLECTION, indent=2), encoding="utf-8")
    print(f"wrote {out}")
