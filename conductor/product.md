# Initial Concept\n\nA CLI tool to convert university regulation HWP files into structured JSON, supporting local/cloud LLMs for preprocessing.
# Product Guide

## Initial Concept
A CLI tool to convert university regulation HWP files into structured JSON, supporting local/cloud LLMs for preprocessing.

## Target Audience
*   **University Administrative Staff**: Individuals responsible for managing and maintaining regulation files who need an automated way to process them.
*   **System Developers**: Engineers who need to integrate the structured regulation data into other university services or websites.

## Core Goals
*   **Automation**: Eliminate the manual effort required to convert HWP files into a digital-friendly format.
*   **Structured Database**: Produce a high-quality, queryable database of regulations from unstructured documents.
*   **Accuracy**: Achieve high fidelity in text extraction and structural parsing, utilizing LLMs to handle complex or low-quality source files.

## Key Features
*   **CLI Interface**: A robust command-line tool for efficient batch processing of regulation files.
*   **Hybrid Preprocessing**: A flexible engine that utilizes Regex for speed and LLMs (Local/Cloud) for handling complex cases.
*   **Hierarchical JSON Output**: Generates structured data broken down by Article, Paragraph, and Item, ensuring logical consistency.
*   **High-Fidelity Tables**: Preserves complex table structures in appendices as raw HTML for perfect visual representation.
*   **Cross-Reference Extraction**: Automatically identifies and links internal regulation references (e.g., "See Article 5").

## Non-Functional Requirements
*   **Privacy & Local Execution**: Capable of running entirely locally using models like Ollama or LM Studio to ensure sensitive data does not leave the network.
*   **Modern Python Stack**: Built on Python 3.11+ and managed with `uv` for easy deployment and dependency management.
*   **Schema Compliance**: Output must strictly adhere to the defined JSON schema to ensure compatibility with downstream systems.

## User Interface
*   **Command Line Interface (CLI)**: The primary interaction model is a terminal-based interface, focusing on utility and scriptability.

