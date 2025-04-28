class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register(self, model_name, version, model_obj):
        self.models[(model_name, version)] = model_obj

    def get_model(self, model_name, version):
        return self.models.get((model_name, version))