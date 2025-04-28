def liveness():
    return {"status": "alive"}

def readiness():
    # TODO: Bağımlılık kontrolleri
    return {"status": "ready"}