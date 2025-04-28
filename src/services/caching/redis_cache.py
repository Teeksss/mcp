import redis

class RedisCache:
    def __init__(self, url):
        self.redis = redis.Redis.from_url(url)

    def set(self, key, value, ttl=None):
        self.redis.set(key, value, ex=ttl)

    def get(self, key):
        return self.redis.get(key)