import httpx
from fastapi import HTTPException

async def fetch_account_details(account_name: str, environment: str) -> dict:
    """
    Call your real backend API here. Returns a JSON dict.
    """
    # Example:
    # url = f"https://api.yourdomain.com/accounts/{account_name}?env={environment}"
    # resp = await httpx.get(url)
    # if resp.status_code != 200:
    #     raise HTTPException(status_code=resp.status_code, detail=resp.text)
    # return resp.json()

    # Stubbed data for demo:
    return {
        "accountName": account_name,
        "environment": environment,
        "balance": 1234.56,
        "ips": ["192.168.1.1", "10.0.0.5"],
        "owner": "John Doe",
        "createdAt": "2023-11-01T10:00:00Z"
    }
