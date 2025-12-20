# Product Guidelines

## Tone and Style
*   **Informative and Helpful**: The CLI output and documentation should provide clear context, actionable suggestions, and detailed explanations for any errors encountered. Avoid overly terse responses; aim to guide the user through the process.

## Coding Standards
*   **PEP 8 Compliance**: Strictly adhere to PEP 8 style guidelines for all Python code to ensure readability and maintainability.
*   **Type Hinting**: Utilize the `typing` module to provide explicit type hints for all function arguments and return values. This enhances code clarity and enables static analysis.
*   **Modular Architecture**: Design the system with a strong emphasis on separation of concerns. Distinct modules should handle preprocessing, parsing, LLM interaction, and JSON output generation.

## Error Handling
*   **Graceful Degradation**: The system should be robust enough to handle failures in individual files without crashing the entire batch process. Skip problematic files and continue processing others.
*   **Detailed Logging**: Maintain a comprehensive log file that captures debugging information, warnings, and errors to assist in troubleshooting.
*   **User-Friendly Feedback**: Present error messages on the CLI in plain language, avoiding raw stack traces unless the user explicitly enables a verbose/debug mode.

## Version Control
*   **Conventional Commits**: Follow the Conventional Commits specification (e.g., `feat:`, `fix:`, `docs:`, `chore:`) for all commit messages to automate changelog generation and semantic versioning.

## Documentation
*   **Comprehensive README**: Maintain an up-to-date `README.md` that serves as the single source of truth for installation, configuration, and usage.
*   **Inline Documentation**: Include thorough docstrings for all public classes, methods, and functions, explaining their purpose, arguments, and return values.
*   **Schema Reference**: Keep `SCHEMA_REFERENCE.md` synchronized with the actual JSON output structure to serve as a reliable reference for downstream consumers.

