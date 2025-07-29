import re

def parse_log_line(log: str) -> dict:
    pattern = r'(\S+) - - \[(.*?)\] "(\S+) (\S+) \S+" (\d{3}) (\d+) "(.*?)" "(.*?)"'
    match = re.match(pattern, log)
    if not match:
        raise ValueError(f"Invalid log format: {log}")

    ip, timestamp, method, url, status, size, referer, user_agent = match.groups()
    return {
        "method": method,
        "url": url,
        "status": int(status),
        "size": int(size),
        "referer": referer,
        "user_agent": user_agent,
    }
