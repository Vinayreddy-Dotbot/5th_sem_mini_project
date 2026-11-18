"""Password hashing for user login (Python standard library only).

Passwords are never stored. We store a salted PBKDF2-SHA256 hash in the form
    pbkdf2_sha256$<iterations>$<salt hex>$<hash hex>
"""
import hashlib
import hmac
import secrets

ITERATIONS = 600_000  # OWASP recommendation for PBKDF2-HMAC-SHA256


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, ITERATIONS)
    return f'pbkdf2_sha256${ITERATIONS}${salt.hex()}${digest.hex()}'


def verify_password(password, stored_hash):
    if not stored_hash:
        return False
    try:
        _, iterations, salt_hex, digest_hex = stored_hash.split('$')
        digest = hashlib.pbkdf2_hmac('sha256', password.encode(),
                                     bytes.fromhex(salt_hex), int(iterations))
    except ValueError:
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)
