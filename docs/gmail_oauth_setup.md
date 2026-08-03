# Gmail OAuth setup for GitHub Actions

This is a one-time setup. Never commit the downloaded OAuth client JSON, `.env.gmail`, or any refresh token.

## 1. Create a Google Cloud project

1. Open Google Cloud Console.
2. Create or select a project dedicated to `Research Inbox`.
3. Enable the **Gmail API** for that project.

## 2. Configure Google Auth Platform

1. Open **Google Auth Platform**.
2. Under **Branding**, set an app name such as `Research Inbox`, a support email, and a contact email.
3. Under **Audience**, choose **External** for a personal Gmail account.
4. During initial setup, add the Gmail account that receives Scholar alerts as a test user.
5. Under **Data Access**, add only:

   `https://www.googleapis.com/auth/gmail.readonly`

6. Before generating the final long-lived refresh token, move the app from **Testing** to **In production**. Testing-mode refresh tokens for non-basic scopes expire after seven days.

For personal use, the app can remain unverified. Google may show an unverified-app warning during consent. Do not authorize any account other than the intended Scholar-alert Gmail account.

## 3. Create a desktop OAuth client

1. Under **Clients**, choose **Create Client**.
2. Select **Desktop app**.
3. Name it `Research Inbox Local Setup`.
4. Download the OAuth client JSON to your own computer.

Do not upload this JSON to GitHub or this chat.

## 4. Generate the refresh token locally

Clone the repository and check out the project branch after it has been merged, then run:

```bash
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the project and run the setup helper:

```bash
pip install -e .
python -m scripts.auth.gmail_oauth_setup /path/to/client_secret.json
```

A browser window will request read-only Gmail access. After consent, the helper writes `.env.gmail`. That file is ignored by Git and must remain local.

## 5. Add repository secrets

Open the GitHub repository and go to:

`Settings → Secrets and variables → Actions → New repository secret`

Create these three repository secrets from `.env.gmail`:

- `GMAIL_CLIENT_ID`
- `GMAIL_CLIENT_SECRET`
- `GMAIL_REFRESH_TOKEN`

Do not create a secret for the temporary access token.

After the secrets are saved, delete the downloaded OAuth client JSON and `.env.gmail` from any shared or synchronized folder. Keeping an encrypted local backup is optional.

## 6. Activate and test

1. Merge the initialization PR into `main`.
2. Open **Actions → Daily Research Inbox**.
3. Choose **Run workflow**.
4. Confirm that the run succeeds.
5. Check the `automation-state` branch:
   - `state/gmail_ingestion_state.json` should have a successful timestamp.
   - `data/paper_registry.jsonl` should contain normalized candidates when new Scholar alerts exist.

The workflow runs at 08:17 and 20:47 Singapore time. It reads only Google Scholar alert messages, stores no raw MIME, and never prints email bodies or paper titles to Actions logs.
