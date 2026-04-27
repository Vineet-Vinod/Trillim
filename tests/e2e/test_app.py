from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from trillim.components import Component
from trillim.server import Server


class E2EServerTests(unittest.TestCase):
    def test_server_serves_health_endpoint_through_real_fastapi_app(self):
        server = Server(Component())

        with TestClient(server.app) as client:
            response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
