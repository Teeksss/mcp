from fastapi import HTTPException, Header

PERMISSIONS = [
    "upload_document",
    "delete_document",
    "manage_models",
    "view_metrics",
    "import_from_drive"
]

def check_permission(token: str = Header(...)):
    # Dummy: Token decode ve permission check
    if token == "dummy-token":
        return True
    raise HTTPException(status_code=403, detail="Yetkisiz")