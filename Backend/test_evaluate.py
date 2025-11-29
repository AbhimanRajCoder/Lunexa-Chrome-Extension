import unittest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
import sys
import os
import asyncio

# Add Backend to sys.path to import chat
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chat import router

# Create a simple app for testing the router
from fastapi import FastAPI
app = FastAPI()
app.include_router(router)

class TestEvaluateEndpoint(unittest.TestCase):
    def test_evaluate_endpoint(self):
        async def run_test():
            # Mock the return value of evaluate_with_openai
            mock_scores = {
                "hallucination": 0.1,
                "consistency": 0.9,
                "fake_news_probability": 0.2,
                "factual_accuracy": 0.8
            }

            # Patch evaluate_with_openai in chat module
            # Note: We need to patch where it is IMPORTED or DEFINED. 
            # Since it is defined in chat.py, we patch 'chat.evaluate_with_openai'.
            with patch("chat.evaluate_with_openai", new_callable=AsyncMock) as mock_eval:
                mock_eval.return_value = mock_scores

                payload = {
                    "query": "What is the capital of France?",
                    "response": "The capital of France is Paris."
                }

                # Use AsyncClient to call the app
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post("/evaluate", json=payload)

                self.assertEqual(response.status_code, 200)
                data = response.json()

                # Check fields
                self.assertIn("cars", data)
                self.assertIn("consistency", data)
                self.assertIn("fake_news_probability", data)
                self.assertIn("factual_accuracy", data)
                self.assertIn("details", data)

                # Check values
                # OpenAI: 0.8 factual
                # Dummy: deterministic based on input.
                # We can just assert ranges or check that values are combined.
                
                print(f"Response data: {data}")
                
                self.assertTrue(0.0 <= data["cars"] <= 1.0)
                self.assertTrue(0.0 <= data["consistency"] <= 1.0)
                self.assertTrue(0.0 <= data["fake_news_probability"] <= 1.0)
                self.assertTrue(0.0 <= data["factual_accuracy"] <= 1.0)

                # Check details
                self.assertEqual(data["details"]["query"], payload["query"])
                self.assertEqual(data["details"]["response"], payload["response"])
                self.assertEqual(data["details"]["openai_raw"], mock_scores)
                self.assertIn("grounded_raw", data["details"])

        # Run the async test function
        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()
