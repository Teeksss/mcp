import smtplib
import requests

ALARM_RULES = [
    {"metric": "latency", "module": "GPT-4", "threshold": 2.0, "type": "critical"},
    {"metric": "error_rate", "module": "RAG", "threshold": 0.1, "type": "warning"},
]

NOTIFY_EMAIL = "alert@example.com"
SLACK_WEBHOOK = "https://hooks.slack.com/services/T000/B000/XXXX"

def send_email(subject: str, text: str):
    with smtplib.SMTP("smtp.example.com") as s:
        s.sendmail("monitor@example.com", [NOTIFY_EMAIL], f"Subject: {subject}\n\n{text}")

def send_slack(message: str):
    requests.post(SLACK_WEBHOOK, json={"text": message})

def check_and_trigger_alarms(metrics: dict):
    for rule in ALARM_RULES:
        val = metrics.get(rule["module"], {}).get(rule["metric"], 0)
        if val > rule["threshold"]:
            msg = f"{rule['module']} için {rule['metric']} alarmı: {val} > {rule['threshold']}"
            send_email("MCP Alarm", msg)
            send_slack(msg)
            # Webhook örneği:
            # requests.post("https://webhook.site/...", json={"msg": msg})