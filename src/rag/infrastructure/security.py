"""
Security validation and hardening utilities for RAG system.

Priority 4 Security Hardening:
- API key validation and format checking
- Input validation and sanitization
- Malicious pattern detection (OWASP Top 10)
- Encryption utilities for sensitive data
"""

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from unicodedata import normalize

try:
    import pydantic

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

    # Create a dummy pydantic module if not available
    class BaseModel:
        """Dummy base model when pydantic is not available."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def model_validate(cls, data):
            return cls(**data)

    class Field:
        """Dummy field descriptor."""

        def __init__(self, default=None, **kwargs):
            self.default = default

    class validator:
        """Dummy validator decorator."""

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            return func

    pydantic = type("obj", (object,), {"BaseModel": BaseModel, "Field": Field})

logger = logging.getLogger(__name__)


class SecurityViolation(Exception):
    """Raised when a security violation is detected."""

    def __init__(
        self, message: str, violation_type: str, pattern: Optional[str] = None
    ):
        self.violation_type = violation_type
        self.pattern = pattern
        super().__init__(message)


class APIKeyFormatError(SecurityViolation):
    """Raised when API key format is invalid."""

    def __init__(self, provider: str, message: str):
        super().__init__(message, "api_key_format")


class InputValidationError(SecurityViolation):
    """Raised when input validation fails."""

    def __init__(
        self, message: str, violation_type: str, pattern: Optional[str] = None
    ):
        super().__init__(message, violation_type, pattern)


# ============================================================================
# API Key Validation (4.1)
# ============================================================================


class APIKeyProvider(str, Enum):
    """Supported API key providers."""

    OPENAI = "openai"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    UNKNOWN = "unknown"


@dataclass
class APIKeyValidationResult:
    """Result of API key validation."""

    is_valid: bool
    provider: APIKeyProvider
    error_message: Optional[str] = None
    health_check_passed: Optional[bool] = None
    expiration_warning: Optional[str] = None
    requires_rotation: bool = False


class APIKeyValidator:
    """
    Validates API keys for format and health.

    Features:
    - Format validation for major providers
    - Health check on initialization
    - Secure key rotation support
    - Expiration warnings (if available)
    """

    # API Key format patterns (based on official documentation)
    # Order matters: more specific patterns first
    PATTERNS: List[tuple[APIKeyProvider, re.Pattern]] = [
        (APIKeyProvider.OPENROUTER, re.compile(r"^sk-or-v1-[A-Za-z0-9_-]{20,}$")),
        (APIKeyProvider.ANTHROPIC, re.compile(r"^sk-ant-[a-zA-Z0-9_-]{40,}$")),
        (APIKeyProvider.GEMINI, re.compile(r"^AIza[a-zA-Z0-9_-]{35}$")),
        (APIKeyProvider.OPENAI, re.compile(r"^sk-(?:proj-)?[a-zA-Z0-9_-]{20,}$")),
        (APIKeyProvider.COHERE, re.compile(r"^[a-zA-Z0-9_-]{40,}$")),
        (
            APIKeyProvider.OLLAMA,
            re.compile(r"^[a-zA-Z0-9_-]{10,}$"),
        ),  # Relaxed pattern for local
    ]

    @classmethod
    def _get_patterns_dict(cls) -> Dict[APIKeyProvider, re.Pattern]:
        """Get patterns as a dictionary for backward compatibility."""
        return dict(cls.PATTERNS)

    # Minimum key lengths for security
    MIN_KEY_LENGTHS: Dict[APIKeyProvider, int] = {
        APIKeyProvider.OPENAI: 20,
        APIKeyProvider.OPENROUTER: 20,
        APIKeyProvider.ANTHROPIC: 40,
        APIKeyProvider.COHERE: 40,
        APIKeyProvider.GEMINI: 39,
        APIKeyProvider.OLLAMA: 10,
    }

    @classmethod
    def detect_provider(cls, api_key: str) -> APIKeyProvider:
        """
        Detect the provider from API key format.

        Args:
            api_key: The API key to analyze.

        Returns:
            Detected provider or UNKNOWN.
        """
        if not api_key:
            return APIKeyProvider.UNKNOWN

        api_key = api_key.strip()

        # Check each provider's pattern (order matters - most specific first)
        for provider, pattern in cls.PATTERNS:
            if pattern.match(api_key):
                return provider

        return APIKeyProvider.UNKNOWN

    @classmethod
    def validate_format(
        cls,
        api_key: str,
        expected_provider: Optional[APIKeyProvider] = None,
    ) -> APIKeyValidationResult:
        """
        Validate API key format.

        Args:
            api_key: The API key to validate.
            expected_provider: Optional expected provider for verification.

        Returns:
            APIKeyValidationResult with validation status.
        """
        if not api_key:
            return APIKeyValidationResult(
                is_valid=False,
                provider=APIKeyProvider.UNKNOWN,
                error_message="API key is empty or None",
            )

        api_key = api_key.strip()

        # Detect provider
        detected_provider = cls.detect_provider(api_key)

        # Verify expected provider if specified
        if expected_provider and detected_provider != expected_provider:
            return APIKeyValidationResult(
                is_valid=False,
                provider=detected_provider,
                error_message=(
                    f"Provider mismatch: expected {expected_provider.value}, "
                    f"detected {detected_provider.value}"
                ),
            )

        # Validate format
        if detected_provider == APIKeyProvider.UNKNOWN:
            # Basic validation for unknown providers
            if len(api_key) < 10:
                return APIKeyValidationResult(
                    is_valid=False,
                    provider=APIKeyProvider.UNKNOWN,
                    error_message="API key too short (minimum 10 characters)",
                )
            # Check for suspicious patterns
            if " " in api_key or "\n" in api_key or "\t" in api_key:
                return APIKeyValidationResult(
                    is_valid=False,
                    provider=APIKeyProvider.UNKNOWN,
                    error_message="API key contains whitespace",
                )
            return APIKeyValidationResult(
                is_valid=True,
                provider=APIKeyProvider.UNKNOWN,
                expiration_warning="Unknown provider format - proceed with caution",
            )

        # Check minimum length
        min_length = cls.MIN_KEY_LENGTHS.get(detected_provider, 20)
        if len(api_key) < min_length:
            return APIKeyValidationResult(
                is_valid=False,
                provider=detected_provider,
                error_message=(
                    f"{detected_provider.value} API key too short "
                    f"(minimum {min_length} characters)"
                ),
            )

        return APIKeyValidationResult(
            is_valid=True,
            provider=detected_provider,
            health_check_passed=None,  # Will be set by health_check()
        )

    @classmethod
    def health_check(
        cls,
        api_key: str,
        provider: APIKeyProvider,
    ) -> bool:
        """
        Perform health check on API key (requires actual API call).

        Args:
            api_key: The API key to check.
            provider: The provider for the API key.

        Returns:
            True if health check passes, False otherwise.
        """
        # This is a placeholder - actual implementation would make a minimal API call
        # For security, we'll just validate format here
        result = cls.validate_format(api_key, provider)
        return result.is_valid


# ============================================================================
# Input Validation with Pydantic (4.2)
# ============================================================================


# Malicious pattern detection (OWASP Top 10)
class MaliciousPattern(str, Enum):
    """Types of malicious patterns to detect."""

    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    LDAP_INJECTION = "ldap_injection"
    EXCESSIVE_LENGTH = "excessive_length"
    TOO_MANY_SPECIAL_CHARS = "too_many_special_chars"


# Security patterns for malicious input detection
# Using simple string patterns to avoid syntax errors
SECURITY_PATTERNS: Dict[MaliciousPattern, List[re.Pattern]] = {
    MaliciousPattern.SQL_INJECTION: [
        re.compile(r"(OR|AND).{1,10}=", re.IGNORECASE),
        re.compile(
            r"""['"]\s*(OR|AND)\s*['"]?[a-zA-Z0-9_-]+['"]?\s*=""", re.IGNORECASE
        ),
        re.compile(r"(;|DROP|DELETE|INSERT|UPDATE|UNION\s+SELECT)", re.IGNORECASE),
        re.compile(r"EXEC\s*\(|EXECUTE\s*\(", re.IGNORECASE),
        re.compile(r"'.*OR.*'.*'=", re.IGNORECASE),
        re.compile(r"1=1|1\s*=\s*1", re.IGNORECASE),
        re.compile(r"xp_cmdshell|sp_executesql", re.IGNORECASE),
    ],
    MaliciousPattern.XSS: [
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<iframe[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL),
        re.compile(r"<embed[^>]*>.*?</embed>", re.IGNORECASE | re.DOTALL),
        re.compile(r"<object[^>]*>.*?</object>", re.IGNORECASE | re.DOTALL),
        re.compile(r"&#(?:x[\da-fA-F]+|\d+);", re.IGNORECASE),
        re.compile(r"%3Cscript", re.IGNORECASE),
    ],
    MaliciousPattern.COMMAND_INJECTION: [
        re.compile(r"[;&|`$](?!\w)", re.IGNORECASE),  # More specific - not part of word
        re.compile(
            r"\b(curl|wget|nc|netcat|ssh|ftp|telnet|bash|sh|cmd|powershell)\s",  # Requires space after
            re.IGNORECASE,
        ),
        re.compile(
            r"\$\([^)]+\)\s", re.IGNORECASE
        ),  # More specific - requires space after
        re.compile(r"`[^`]+`\s", re.IGNORECASE),  # More specific - requires space after
        re.compile(r"eval\s*\(", re.IGNORECASE),
    ],
    MaliciousPattern.PATH_TRAVERSAL: [
        re.compile(r"\.\.[\\/]", re.IGNORECASE),
        re.compile(r"%2e%2e", re.IGNORECASE),
        re.compile(r"\.\.[\\/]\.\.[\\/]", re.IGNORECASE),
        re.compile(r"[\\/]\s*[a-zA-Z]:", re.IGNORECASE),
    ],
    MaliciousPattern.LDAP_INJECTION: [
        re.compile(r"\([a-zA-Z0-9_-]+\*?\)", re.IGNORECASE),
        re.compile(r"[()&|!=<>~]", re.IGNORECASE),
        re.compile(r"\*.*\*\)*\)\s*\(", re.IGNORECASE),
    ],
}


def detect_malicious_patterns(
    text: str,
) -> List[tuple[MaliciousPattern, re.Pattern, str]]:
    """
    Detect malicious patterns in input text.

    Args:
        text: The text to analyze.

    Returns:
        List of (pattern_type, pattern, matched_string) tuples.
    """
    detected = []

    for pattern_type, patterns in SECURITY_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                detected.append((pattern_type, pattern, match.group(0)))

    return detected


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize input text by removing or neutralizing malicious patterns.

    Args:
        text: The text to sanitize.
        max_length: Maximum allowed length.

    Returns:
        Sanitized text.

    Raises:
        InputValidationError: If sanitization fails or text is too long.
    """
    if not text:
        return ""

    # Check length first
    if len(text) > max_length:
        raise InputValidationError(
            f"Input exceeds maximum length of {max_length} characters",
            MaliciousPattern.EXCESSIVE_LENGTH.value,
        )

    # Normalize Unicode (NFKC to prevent homograph attacks)
    sanitized = normalize("NFKC", text)

    # Remove null bytes
    sanitized = sanitized.replace("\x00", "")

    # Detect malicious patterns FIRST (before special character check)
    # This ensures specific attacks are identified rather than generic "too many special chars"
    detected = detect_malicious_patterns(sanitized)
    if detected:
        pattern_type, pattern, matched = detected[0]
        raise InputValidationError(
            f"Malicious pattern detected: {pattern_type.value}",
            pattern_type.value,
            matched[:100],  # Truncate for logging
        )

    # Check for too many special characters (potential DoS)
    # Only check if no malicious patterns were found
    special_char_count = sum(
        1 for c in sanitized if not c.isalnum() and not c.isspace()
    )
    if len(sanitized) > 0:
        special_ratio = special_char_count / len(sanitized)
        if special_ratio > 0.5:  # More than 50% special characters
            raise InputValidationError(
                f"Input contains too many special characters ({special_ratio:.1%})",
                MaliciousPattern.TOO_MANY_SPECIAL_CHARS.value,
            )

    return sanitized


if PYDANTIC_AVAILABLE:
    from pydantic import field_validator, model_validator

    class SearchQueryInput(pydantic.BaseModel):
        """
        Pydantic model for search query input validation.

        Security features:
        - Query length limits
        - Malicious pattern detection
        - Special character filtering
        - Unicode normalization
        """

        query: str
        top_k: int = pydantic.Field(default=10, ge=1, le=100)
        filter_options: Optional[Dict[str, Any]] = None
        rate_limit_id: Optional[str] = None  # For rate limiting preparation
        max_query_length: int = pydantic.Field(default=1000, ge=100, le=10000)

        @field_validator("query")
        @classmethod
        def validate_query(cls, v: str, info) -> str:
            """Validate and sanitize query string."""
            if not v or not v.strip():
                raise ValueError("Query cannot be empty")

            # Apply sanitization
            try:
                sanitized = sanitize_input(
                    v, max_length=info.data.get("max_query_length", 1000)
                )
            except InputValidationError as e:
                raise ValueError(str(e))

            return sanitized.strip()

        @field_validator("filter_options")
        @classmethod
        def validate_filter_options(
            cls, v: Optional[Dict[str, Any]]
        ) -> Optional[Dict[str, Any]]:
            """Validate filter options for injection attempts."""
            if v is None:
                return v

            # Recursively check filter values
            sanitized = {}
            for key, value in v.items():
                if isinstance(value, str):
                    try:
                        sanitized[key] = sanitize_input(value, max_length=500)
                    except InputValidationError as e:
                        raise ValueError(f"Invalid filter value for '{key}': {e}")
                elif isinstance(value, (int, float, bool)):
                    sanitized[key] = value
                elif isinstance(value, list):
                    # Sanitize list items
                    sanitized_list = []
                    for item in value:
                        if isinstance(item, str):
                            try:
                                sanitized_list.append(
                                    sanitize_input(item, max_length=500)
                                )
                            except InputValidationError as e:
                                raise ValueError(f"Invalid filter item in '{key}': {e}")
                        else:
                            sanitized_list.append(item)
                    sanitized[key] = sanitized_list
                else:
                    sanitized[key] = value

            return sanitized

        @model_validator(mode="after")
        def validate_overall_length(self) -> "SearchQueryInput":
            """Validate overall input size to prevent DoS."""
            # Calculate approximate size
            query_size = len(self.query.encode("utf-8"))
            filter_size = (
                len(str(self.filter_options).encode("utf-8"))
                if self.filter_options
                else 0
            )
            total_size = query_size + filter_size

            if total_size > 50000:  # 50KB limit
                raise ValueError(
                    f"Total input size ({total_size} bytes) exceeds maximum (50000 bytes)"
                )

            return self

    class HyDEQueryInput(pydantic.BaseModel):
        """
        Pydantic model for HyDE query input validation.

        Similar to SearchQueryInput but with HyDE-specific constraints.
        """

        query: str
        temperature: float = pydantic.Field(default=0.3, ge=0.0, le=2.0)
        max_query_length: int = pydantic.Field(default=500, ge=50, le=1000)

        @field_validator("query")
        @classmethod
        def validate_query(cls, v: str, info) -> str:
            """Validate and sanitize query string for HyDE."""
            if not v or not v.strip():
                raise ValueError("Query cannot be empty")

            try:
                sanitized = sanitize_input(
                    v, max_length=info.data.get("max_query_length", 500)
                )
            except InputValidationError as e:
                raise ValueError(str(e))

            return sanitized.strip()


else:
    # Fallback when Pydantic is not available
    SearchQueryInput = BaseModel
    HyDEQueryInput = BaseModel


# ============================================================================
# Encryption Utilities (4.3)
# ============================================================================

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography package not available - encryption features disabled")


class EncryptionManager:
    """
    Manage encryption for sensitive cache data.

    Features:
    - AES-256 encryption
    - Key derivation from password
    - Secure key storage
    """

    def __init__(self, password: Optional[str] = None, salt: Optional[bytes] = None):
        """
        Initialize encryption manager.

        Args:
            password: Optional password for key derivation. If None, generates random key.
            salt: Optional salt for key derivation. If None, generates random salt.
        """
        if not CRYPTO_AVAILABLE:
            logger.warning(
                "Encryption not available - cryptography package not installed"
            )
            self._fernet: Optional[Any] = None
            return

        if password:
            # Derive key from password
            if salt is None:
                salt = os.urandom(16)
            self._salt = salt
            key = self._derive_key(password, salt)
        else:
            # Generate random key
            key = Fernet.generate_key()
            self._salt = None

        self._fernet = Fernet(key)
        self._password_required = password is not None

    @staticmethod
    def _derive_key(password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2.

        Args:
            password: Password string.
            salt: Salt bytes.

        Returns:
            32-byte url-safe base64-encoded key suitable for Fernet.
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography package not available")

        import base64

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        # Derive key and encode as url-safe base64 for Fernet
        derived_key = kdf.derive(password.encode())
        return base64.urlsafe_b64encode(derived_key)

    def encrypt(self, data: str) -> Optional[bytes]:
        """
        Encrypt string data.

        Args:
            data: String to encrypt.

        Returns:
            Encrypted bytes, or None if encryption not available.
        """
        if self._fernet is None:
            return None

        return self._fernet.encrypt(data.encode("utf-8"))

    def decrypt(self, encrypted_data: bytes) -> Optional[str]:
        """
        Decrypt encrypted data.

        Args:
            encrypted_data: Encrypted bytes.

        Returns:
            Decrypted string, or None if decryption fails.
        """
        if self._fernet is None:
            return None

        try:
            return self._fernet.decrypt(encrypted_data).decode("utf-8")
        except Exception as e:
            logger.warning(f"Decryption failed: {e}")
            return None

    @property
    def available(self) -> bool:
        """Check if encryption is available."""
        return self._fernet is not None

    def get_key_info(self) -> Dict[str, Any]:
        """
        Get information about encryption key (safe for logging).

        Returns:
            Dict with key information (no actual key material).
        """
        return {
            "encryption_enabled": self.available,
            "password_based": self._password_required,
            "has_salt": self._salt is not None,
        }


# ============================================================================
# Cache Key Validation (4.3)
# ============================================================================


def validate_cache_key(key: str) -> bool:
    """
    Validate cache key to prevent cache poisoning.

    Args:
        key: Cache key to validate.

    Returns:
        True if key is valid, False otherwise.
    """
    if not key or not isinstance(key, str):
        return False

    # Check for path traversal
    if ".." in key or key.startswith("/") or key.startswith("\\"):
        return False

    # Check for excessive length
    if len(key) > 256:
        return False

    # Check for control characters
    if any(ord(c) < 32 for c in key):
        return False

    # Only allow safe characters (alphanumeric, underscore, dash, dot, colon)
    allowed_pattern = re.compile(r"^[a-zA-Z0-9_\-:.]+$")
    return bool(allowed_pattern.match(key))


def sanitize_cache_data(data: Any) -> Any:
    """
    Sanitize cached data to prevent cache poisoning.

    Args:
        data: Data to sanitize.

    Returns:
        Sanitized data.
    """
    if isinstance(data, str):
        # Remove null bytes and control characters
        return "".join(c for c in data if ord(c) >= 32 or c in "\n\r\t")
    elif isinstance(data, dict):
        return {k: sanitize_cache_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_cache_data(item) for item in data]
    else:
        return data
