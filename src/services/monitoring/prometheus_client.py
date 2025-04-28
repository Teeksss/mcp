class PrometheusExporter:
    def export_metric(self, name, value):
        print(f"Exporting metric: {name}={value}")