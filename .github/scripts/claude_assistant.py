#!/usr/bin/env python3
"""Claude Assistant for GitHub Issues.

This script processes GitHub issues and comments mentioning @claude,
analyzes the request, and makes appropriate code changes.
"""

import os
import json
import subprocess
from pathlib import Path

import anthropic


def get_repo_context() -> str:
    """Get relevant repository context for Claude."""
    context_parts = []

    # Read CLAUDE.md if it exists
    claude_md = Path("CLAUDE.md")
    if claude_md.exists():
        context_parts.append(f"## CLAUDE.md\n\n{claude_md.read_text()}")

    # Get file structure
    result = subprocess.run(
        ["find", ".", "-type", "f", "-name", "*.py", "-o", "-name", "*.json"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        files = [
            f for f in result.stdout.strip().split("\n")
            if f and not ".venv" in f and not "__pycache__" in f
        ]
        context_parts.append(f"## Repository Files\n\n{chr(10).join(files)}")

    return "\n\n".join(context_parts)


def read_file(filepath: str) -> str:
    """Read a file's contents."""
    try:
        return Path(filepath).read_text()
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(filepath: str, content: str) -> str:
    """Write content to a file."""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        Path(filepath).write_text(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"


def post_comment(issue_number: str, body: str) -> None:
    """Post a comment on the GitHub issue."""
    subprocess.run(
        ["gh", "issue", "comment", issue_number, "--body", body],
        check=True,
    )


def main():
    # Get environment variables
    issue_number = os.environ.get("ISSUE_NUMBER", "")
    issue_title = os.environ.get("ISSUE_TITLE", "")
    issue_body = os.environ.get("ISSUE_BODY", "")
    comment_body = os.environ.get("COMMENT_BODY", "")
    repo_name = os.environ.get("REPO_NAME", "")

    # Determine the request
    request = comment_body if comment_body else issue_body

    # Get repository context
    repo_context = get_repo_context()

    # Define tools for Claude
    tools = [
        {
            "name": "read_file",
            "description": "Read the contents of a file in the repository",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to read",
                    }
                },
                "required": ["filepath"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file in the repository",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["filepath", "content"],
            },
        },
        {
            "name": "post_response",
            "description": "Post a response comment on the GitHub issue",
            "input_schema": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to post as a comment",
                    }
                },
                "required": ["message"],
            },
        },
    ]

    # Create Claude client
    client = anthropic.Anthropic()

    # System prompt
    system_prompt = f"""You are Claude, an AI assistant helping with a GitHub repository.

Repository: {repo_name}

{repo_context}

You are responding to a GitHub issue. Analyze the request and take appropriate action:
1. If it's a bug fix or feature request, read the relevant files, make the necessary changes, and write the updated files.
2. Always post a response explaining what you did or why you couldn't help.
3. Be concise and professional in your responses.
4. Follow the coding style and patterns already in the repository.
5. If you make changes, they will automatically be committed and a PR will be created.

Issue #{issue_number}: {issue_title}
"""

    messages = [{"role": "user", "content": request}]

    # Run Claude with tool use
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        # Process response
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Check if we need to handle tool calls
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input

                    if tool_name == "read_file":
                        result = read_file(tool_input["filepath"])
                    elif tool_name == "write_file":
                        result = write_file(
                            tool_input["filepath"], tool_input["content"]
                        )
                    elif tool_name == "post_response":
                        post_comment(issue_number, tool_input["message"])
                        result = "Comment posted successfully"
                    else:
                        result = f"Unknown tool: {tool_name}"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})
        else:
            # No more tool calls, we're done
            break

    print("Claude Assistant completed successfully")


if __name__ == "__main__":
    main()
