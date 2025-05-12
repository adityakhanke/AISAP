import httpx
import google.auth
from google.auth.transport.requests import Request
import os


def get_credentials():
    credentials, project_id = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    return credentials.token
def build_endpoint_url(
    region: str,
    project_id: str,
    model_name: str,
    streaming: bool = False,
):
    base_url = f"https://{region}-aiplatform.googleapis.com/v1/"
    project_fragment = f"projects/{project_id}"
    location_fragment = f"locations/{region}"
    specifier = "streamRawPredict" if streaming else "rawPredict"
    model_fragment = f"publishers/mistralai/models/{model_name}"
    url = f"{base_url}{'/'.join([project_fragment, location_fragment, model_fragment])}:{specifier}"
    return url


# Retrieve Google Cloud Project ID and Region from environment variables
project_id = "aisap-458706"
region = "us-central1"


# Retrieve Google Cloud credentials.
access_token = get_credentials()


model = "codestral-2501"
is_streamed = False # Change to True to stream token responses


# Build URL
url = build_endpoint_url(
    project_id=project_id,
    region=region,
    model_name=model,
    streaming=is_streamed
)


# Define query headers
headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/json",
}


# Define POST payload
data = {
    "model": model,
    "messages": [{"role": "user", "content": "Who is the best French painter?"}],
    "stream": is_streamed,
}


print(url)
print(data)
print(headers)

# Make the call
with httpx.Client() as client:
    resp = client.post(url, json=data, headers=headers, timeout=None)
    print(resp.text)