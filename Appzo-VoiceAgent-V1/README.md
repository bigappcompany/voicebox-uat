# Appzo VoiceAgent V1

GoodBox/Plivo telephone voice agent. Configuration remains in the parent
`../.env`; it has intentionally not been copied or changed.

## Run

From this directory, install the dependencies if needed:

```bash
python3 -m pip install -r requirements.txt
```

Start the media endpoint, then expose it on port 8000:

```bash
python3 goodbox_server.py
ngrok http 8000
```

Set `PUBLIC_BASE_URL` in the parent `.env` to the current ngrok HTTPS URL, then
initiate a test call:

```bash
python3 scripts/dial_goodbox_test.py --to +918274828890
```
