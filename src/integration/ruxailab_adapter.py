import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


class RUXAILABAdapter:
    """Sends benchmark reports to a RUXAILAB instance."""

    def __init__(self, endpoint, study_id=None, api_key=None, timeout_s=10.0):
        self.endpoint = endpoint.rstrip("/")
        self.study_id = study_id
        self.api_key = api_key
        self.timeout = timeout_s

    def post_report(self, report_dict):
        payload = json.dumps({
            "type": "benchmark_result",
            "study_id": self.study_id,
            "payload": report_dict,
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.endpoint}/benchmark"
        req = Request(url, data=payload, headers=headers, method="POST")

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                if resp.status in (200, 201):
                    print(f"Report sent to RUXAILAB successfully!")
                    return True
                else:
                    print(f"RUXAILAB returned unexpected status: {resp.status}")
                    return False

        except HTTPError as e:
            print(f"HTTP error when contacting RUXAILAB: {e.code} - {e.reason}")
            return False

        except URLError as e:
            print(f"Could not connect to RUXAILAB at {self.endpoint}")
            print(f"Make sure RUXAILAB is running. Error: {e.reason}")
            return False
