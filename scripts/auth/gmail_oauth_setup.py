from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"
GMAIL_SCOPES = (GMAIL_READONLY_SCOPE, GMAIL_SEND_SCOPE)


def load_client_values(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    client = payload.get("installed") or payload.get("web")
    if not isinstance(client, dict):
        raise ValueError("OAuth client JSON must contain an installed or web client")
    client_id = client.get("client_id")
    client_secret = client.get("client_secret")
    if not client_id or not client_secret:
        raise ValueError("OAuth client JSON is missing client_id or client_secret")
    return str(client_id), str(client_secret)


def shell_value(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a one-time local OAuth consent flow for read-only Gmail access "
            "and outbound email delivery, then write credentials to an ignored "
            "environment file."
        )
    )
    parser.add_argument("client_secret_file", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".env.gmail"),
        help="Ignored local file receiving the three GitHub Secret values",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing credential file: {args.output}"
        )

    client_id, client_secret = load_client_values(args.client_secret_file)
    flow = InstalledAppFlow.from_client_secrets_file(
        str(args.client_secret_file),
        scopes=list(GMAIL_SCOPES),
    )
    credentials = flow.run_local_server(
        port=0,
        open_browser=True,
        access_type="offline",
        prompt="consent",
    )
    if not credentials.refresh_token:
        raise RuntimeError(
            "Google did not return a refresh token; revoke the prior grant and retry"
        )

    granted_scopes = set(credentials.scopes or [])
    missing_scopes = [scope for scope in GMAIL_SCOPES if scope not in granted_scopes]
    if missing_scopes:
        raise RuntimeError(
            "Google authorization did not grant all required Gmail scopes: "
            + ", ".join(missing_scopes)
        )

    content = "\n".join(
        [
            f"GMAIL_CLIENT_ID={shell_value(client_id)}",
            f"GMAIL_CLIENT_SECRET={shell_value(client_secret)}",
            f"GMAIL_REFRESH_TOKEN={shell_value(credentials.refresh_token)}",
            "",
        ]
    )
    args.output.write_text(content, encoding="utf-8")
    try:
        os.chmod(args.output, 0o600)
    except OSError:
        pass

    print(f"Credential values were written to {args.output}.")
    print("Granted Gmail permissions: read-only mailbox access and email sending.")
    print("Do not commit or upload this file. Add each value to GitHub Actions Secrets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
