#!/usr/bin/env python3
"""Upload generated Excel files to OneDrive/SharePoint using Microsoft Graph."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


ROOT = Path(__file__).resolve().parent
DEFAULT_TARGET_URL = (
    "https://listgs-my.sharepoint.com/personal/l_kornblum_list-goslar_com/"
    "_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fl%5Fkornblum%5Flist%2Dgoslar%5Fcom"
    "%2FDocuments%2FAI%2Dbasierte%20Preisgestaltung&ga=1"
)
DEFAULT_FILE = ROOT / "parsed" / "KW19_2026" / "matched_competitor_products.xlsx"
DEFAULT_TOKEN_CACHE = ROOT / ".onedrive_token.json"
GRAPH_ROOT = "https://graph.microsoft.com/v1.0"
DEVICE_CODE_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/devicecode"
TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
SCOPES = "User.Read Files.ReadWrite.All Sites.ReadWrite.All offline_access"
REQUEST_TIMEOUT = (10, 60)
UPLOAD_TIMEOUT = (10, 180)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload an Excel file to the LIST OneDrive folder.")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--target-url", default=DEFAULT_TARGET_URL)
    parser.add_argument("--target-folder", help="Graph folder path override, e.g. /personal/.../Documents/AI-basierte Preisgestaltung")
    parser.add_argument("--tenant", default=os.environ.get("MS_TENANT_ID", "common"))
    parser.add_argument("--client-id", default=os.environ.get("MS_CLIENT_ID"))
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--filename", help="Optional upload filename. Defaults to local file name.")
    return parser


def main() -> None:
    if load_dotenv:
        load_dotenv(ROOT / ".env", override=False)
    args = build_parser().parse_args()
    upload_to_onedrive(
        file_path=args.file,
        target_url=args.target_url,
        target_folder=args.target_folder,
        tenant=args.tenant,
        client_id=args.client_id or os.environ.get("MS_CLIENT_ID", ""),
        token_cache=args.token_cache,
        filename=args.filename,
    )


def upload_to_onedrive(
    file_path: Path,
    target_url: str = DEFAULT_TARGET_URL,
    target_folder: str | None = None,
    tenant: str = "common",
    client_id: str = "",
    token_cache: Path = DEFAULT_TOKEN_CACHE,
    filename: str | None = None,
) -> str:
    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Upload file not found: {file_path}")
    if not client_id:
        raise RuntimeError(
            "MS_CLIENT_ID is missing. Create an Azure app registration with public-client/device-code "
            "flow enabled, then add MS_CLIENT_ID to .env."
        )

    folder_path = target_folder or folder_path_from_onedrive_url(target_url)
    upload_name = filename or file_path.name
    print(f"Upload file: {file_path}")
    print(f"Target folder: {folder_path or 'shared link'}")
    token = get_access_token(client_id, tenant, token_cache)
    signed_in_user = get_signed_in_user(token)
    if signed_in_user:
        print(f"Signed in Microsoft account: {signed_in_user}")
    target_domain = target_domain_from_personal_folder(folder_path)

    print(f"Uploading as: {upload_name}")
    if is_sharepoint_sharing_link(target_url) or should_use_shared_folder_route(folder_path, target_domain, signed_in_user):
        print("Using shared-folder upload route for external OneDrive access.")
        upload_url = shared_folder_upload_url(target_url, upload_name, token)
        response = upload_file_content(file_path, upload_url, token)
    else:
        upload_url = graph_upload_url(target_url, folder_path, upload_name)
        response = upload_file_content(file_path, upload_url, token)

    if response.status_code >= 400 and parsed_sharepoint_url(target_url):
        print("Direct site upload failed; trying shared-folder resolution.")
        shared_upload_url = shared_folder_upload_url(target_url, upload_name, token)
        response = upload_file_content(file_path, shared_upload_url, token)
    if response.status_code >= 400:
        raise RuntimeError(f"OneDrive upload failed: {response.status_code} {response.text}")

    payload = response.json()
    web_url = payload.get("webUrl", "")
    print(f"Uploaded {file_path.name} to OneDrive: {web_url or upload_name}")
    return web_url


def upload_file_content(file_path: Path, upload_url: str, token: str) -> requests.Response:
    with file_path.open("rb") as handle:
        return requests.put(
            upload_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            },
            data=handle,
            timeout=UPLOAD_TIMEOUT,
        )


def parsed_sharepoint_url(target_url: str) -> bool:
    parsed = urlparse(target_url)
    return bool(parsed.hostname and parsed.hostname.endswith(".sharepoint.com"))


def should_use_shared_folder_route(folder_path: str, target_domain: str, signed_in_user: str) -> bool:
    if not is_personal_onedrive_folder(folder_path):
        return False
    if not target_domain:
        return True
    if not signed_in_user:
        return True
    return not signed_in_user.lower().endswith(f"@{target_domain}")


def shared_folder_upload_url(target_url: str, upload_name: str, token: str) -> str:
    share_id = encode_sharing_url(target_url)
    response = requests.get(
        f"{GRAPH_ROOT}/shares/{share_id}/driveItem",
        headers={"Authorization": f"Bearer {token}"},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        print(f"Shared-link resolution failed: {response.status_code} {response.text}")
        return shared_with_me_upload_url(target_url, upload_name, token)
    item = response.json()
    return upload_url_for_drive_item(item, upload_name)


def shared_with_me_upload_url(target_url: str, upload_name: str, token: str) -> str:
    expected_path = folder_path_from_onedrive_url(target_url)
    expected_folder = Path(expected_path).name.casefold() if expected_path else ""
    response = requests.get(
        f"{GRAPH_ROOT}/me/drive/sharedWithMe",
        headers={"Authorization": f"Bearer {token}"},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Could not list shared OneDrive items: {response.status_code} {response.text}")
    items = response.json().get("value") or []
    matches = []
    if expected_folder:
        matches = [
            item for item in items
            if str(item.get("name") or "").casefold() == expected_folder
        ]
    if not matches:
        available = ", ".join(str(item.get("name") or "") for item in items[:20])
        raise RuntimeError(
            f"Shared folder '{expected_folder}' was not found in sharedWithMe. "
            f"Available shared items: {available or 'none'}"
        )
    return upload_url_for_drive_item(matches[0].get("remoteItem") or matches[0], upload_name)


def upload_url_for_drive_item(item: dict[str, object], upload_name: str) -> str:
    drive_id = (item.get("parentReference") or {}).get("driveId")
    item_id = item.get("id")
    if not drive_id or not item_id:
        raise RuntimeError(f"Resolved OneDrive item is missing drive id or item id: {item}")
    return f"{GRAPH_ROOT}/drives/{drive_id}/items/{item_id}:/{quote_segment(upload_name)}:/content"


def encode_sharing_url(url: str) -> str:
    encoded = base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii").rstrip("=")
    return f"u!{encoded}"


def graph_upload_url(target_url: str, folder_path: str, upload_name: str) -> str:
    parsed = urlparse(target_url)
    if is_personal_onedrive_folder(folder_path):
        _, drive_folder = split_personal_onedrive_path(folder_path)
        return f"{GRAPH_ROOT}/me/drive/root:{quote_graph_path(drive_folder)}/{quote_segment(upload_name)}:/content"
    if parsed.hostname and parsed.hostname.endswith(".sharepoint.com"):
        site_path, drive_folder = split_personal_onedrive_path(folder_path)
        return (
            f"{GRAPH_ROOT}/sites/{parsed.hostname}:{quote_graph_path(site_path)}:"
            f"/drive/root:{quote_graph_path(drive_folder)}/{quote_segment(upload_name)}:/content"
        )
    return f"{GRAPH_ROOT}/me/drive/root:{quote_graph_path(folder_path)}/{quote_segment(upload_name)}:/content"


def is_personal_onedrive_folder(folder_path: str) -> bool:
    parts = [part for part in folder_path.strip("/").split("/") if part]
    return len(parts) >= 3 and parts[0] == "personal" and parts[2].lower() == "documents"


def split_personal_onedrive_path(folder_path: str) -> tuple[str, str]:
    parts = [part for part in folder_path.strip("/").split("/") if part]
    if is_personal_onedrive_folder(folder_path):
        site_path = "/" + "/".join(parts[:2])
        drive_folder = "/" + "/".join(parts[3:])
        return site_path, drive_folder
    return folder_path, "/"


def get_signed_in_user(token: str) -> str:
    response = requests.get(
        f"{GRAPH_ROOT}/me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        print(f"Could not read signed-in Microsoft user: {response.status_code} {response.text}")
        return ""
    user = response.json()
    return user.get("userPrincipalName") or user.get("mail") or user.get("displayName") or ""


def target_domain_from_personal_folder(folder_path: str) -> str:
    parts = [part for part in folder_path.strip("/").split("/") if part]
    if len(parts) < 2 or parts[0] != "personal":
        return ""
    owner = parts[1]
    marker = "_list-goslar_com"
    if owner.endswith(marker):
        return "list-goslar.com"
    return ""


def folder_path_from_onedrive_url(url: str) -> str:
    query = parse_qs(urlparse(url).query)
    raw_id = (query.get("id") or [""])[0]
    folder = unquote(raw_id)
    if not folder and is_sharepoint_sharing_link(url):
        return ""
    if not folder:
        raise RuntimeError("Could not extract target folder from OneDrive URL. Use --target-folder.")
    return folder


def is_sharepoint_sharing_link(url: str) -> bool:
    parsed = urlparse(url)
    return bool(
        parsed.hostname
        and parsed.hostname.endswith(".sharepoint.com")
        and ("/:f:/" in parsed.path or "/:x:/" in parsed.path or "/:w:/" in parsed.path)
    )


def quote_graph_path(path: str) -> str:
    parts = [quote_segment(part) for part in path.strip("/").split("/") if part]
    return "/" + "/".join(parts)


def quote_segment(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


def get_access_token(client_id: str, tenant: str, token_cache: Path) -> str:
    cached = read_token_cache(token_cache)
    if cached and cached.get("access_token") and cached.get("expires_at", 0) > time.time() + 120:
        print("Using cached Microsoft Graph access token.")
        return cached["access_token"]
    if cached and cached.get("refresh_token"):
        print("Refreshing Microsoft Graph access token.")
        refreshed = refresh_access_token(client_id, tenant, cached["refresh_token"])
        write_token_cache(token_cache, refreshed)
        return refreshed["access_token"]
    print("Requesting Microsoft device-code login.")
    token = device_code_login(client_id, tenant)
    write_token_cache(token_cache, token)
    return token["access_token"]


def read_token_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_token_cache(path: Path, token: dict) -> None:
    path.write_text(json.dumps(token, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except Exception:
        pass


def normalize_token_payload(payload: dict) -> dict:
    expires_in = int(payload.get("expires_in") or 3600)
    payload["expires_at"] = time.time() + expires_in
    return payload


def refresh_access_token(client_id: str, tenant: str, refresh_token: str) -> dict:
    response = requests.post(
        TOKEN_URL.format(tenant=tenant),
        data={
            "client_id": client_id,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "scope": SCOPES,
        },
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        return device_code_login(client_id, tenant)
    return normalize_token_payload(response.json())


def device_code_login(client_id: str, tenant: str) -> dict:
    print(f"Requesting device code from Microsoft tenant '{tenant}'.")
    response = requests.post(
        DEVICE_CODE_URL.format(tenant=tenant),
        data={"client_id": client_id, "scope": SCOPES},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Device code request failed: {response.status_code} {response.text}")
    device = response.json()
    print(device.get("message") or f"Open {device['verification_uri']} and enter code {device['user_code']}")

    deadline = time.time() + int(device.get("expires_in") or 900)
    interval = int(device.get("interval") or 5)
    while time.time() < deadline:
        time.sleep(interval)
        token_response = requests.post(
            TOKEN_URL.format(tenant=tenant),
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": client_id,
                "device_code": device["device_code"],
            },
            timeout=REQUEST_TIMEOUT,
        )
        payload = token_response.json()
        if token_response.status_code == 200:
            return normalize_token_payload(payload)
        error = payload.get("error")
        if error in {"authorization_pending", "slow_down"}:
            if error == "slow_down":
                interval += 5
            continue
        raise RuntimeError(f"Device code login failed: {token_response.status_code} {token_response.text}")
    raise RuntimeError("Device code login timed out.")


if __name__ == "__main__":
    main()
