from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("request_count", "Total API Requests", ["endpoint"])
REQUEST_TIME = Histogram("request_time_seconds", "API Request Time", ["endpoint"])