import ldap3
from ldap3.core.exceptions import LDAPException

LDAP_SERVER = "ldap://your-ldap-server.domain"
BASE_DN = "ou=users,dc=yourdomain,dc=com"

def ldap_authenticate(username: str, password: str) -> bool:
    try:
        server = ldap3.Server(LDAP_SERVER, get_info=ldap3.ALL)
        user_dn = f"uid={username},{BASE_DN}"
        conn = ldap3.Connection(server, user=user_dn, password=password, auto_bind=True)
        return conn.bound
    except LDAPException:
        return False