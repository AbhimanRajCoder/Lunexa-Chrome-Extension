import unittest
from httpx import AsyncClient, ASGITransport
import sys
import os
import asyncio

# Add Backend to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import app

class TestExtensionEndpoint(unittest.TestCase):
    def test_score_endpoint(self):
        async def run_test():
            payload = {
                "query": "Is the earth flat?",
                "response": "The earth is an oblate spheroid."
            }
            
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/score", json=payload)
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            self.assertIn("scores", data)
            self.assertIn("CARS", data["scores"])
            self.assertIn("Trust_Confidence_Score", data["scores"])
            
            print(f"Extension Score Data: {data}")

        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()
