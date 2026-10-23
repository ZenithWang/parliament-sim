"""Central redaction for provider errors and connection strings."""
import os
import re


def safe_error(error):
    message = str(error)
    for name in ('DEEPSEEK_API_KEY', 'DASHSCOPE_API_KEY', 'REDIS_URL'):
        value = os.getenv(name)
        if value:
            message = message.replace(value, '[REDACTED]')
    message = re.sub(r'(rediss?://)[^\s/@]+@', r'\1[REDACTED]@', message)
    return f'{type(error).__name__}: {message}'
