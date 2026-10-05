"""
Publish this repository to GitHub.

Three environment problems were measured on this machine and are handled here rather than
left to the user to rediscover:

  1. GitHub is only reachable through the local proxy at 127.0.0.1:7897. Python picks that
     up from the Windows registry; git does not, so git needs http.proxy set explicitly.
  2. Windows' own TLS backend cannot acquire a client credential in this session
     (schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS). git's bundled
     OpenSSL backend works, so http.sslBackend=openssl is required.
  3. The sandbox blocks named pipes, which breaks git's credential-helper shell
     ("couldn't create signal pipe, Win32 error 5"). So the token is NOT given to a
     credential helper: it travels in an Authorization header injected through git's
     GIT_CONFIG_KEY_n/GIT_CONFIG_VALUE_n environment variables, which keeps it out of the
     command line, out of .git/config and out of every file.

The token is read from .secrets/github/access_token, which .gitignore covers. It is never
printed, logged or written anywhere.

Usage:
  python scripts/push_github.py --check-only   # validate the token and the remote
  python scripts/push_github.py                # check, then push main
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_PATH = os.path.join(ROOT, ".secrets", "github", "access_token")
OWNER = "leesirleesirlee-star"
REPO = "Enveda_CASMI-2026_Molecule-ID-From-Mass-Spectra"
REMOTE = f"https://github.com/{OWNER}/{REPO}.git"
PROXY = "http://127.0.0.1:7897"
VALID_PREFIXES = ("ghp_", "github_pat_", "gho_", "ghu_", "ghs_")


def git(*args, env=None, check=False):
    e = dict(os.environ)
    e.setdefault("GIT_TERMINAL_PROMPT", "0")
    if env:
        e.update(env)
    p = subprocess.run(["git", "-c", "core.quotePath=false", *args],
                       cwd=ROOT, capture_output=True, env=e)
    out = p.stdout.decode("utf-8", "replace")
    err = p.stderr.decode("utf-8", "replace")
    if check and p.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{err.strip()}")
    return p.returncode, out, err


def read_token():
    if not os.path.exists(TOKEN_PATH):
        raise SystemExit(
            f"No token at {TOKEN_PATH}\n"
            "Create a GitHub personal access token and save it there (see the README steps)."
        )
    tok = open(TOKEN_PATH, encoding="utf-8").read().strip()
    if not tok:
        raise SystemExit("The token file is empty.")
    if any(ch.isspace() for ch in tok):
        raise SystemExit("The token file contains whitespace; it must hold the token alone.")
    if not tok.startswith(VALID_PREFIXES):
        raise SystemExit(
            f"The token starts with {tok[:12]!r}, which is not a recognised GitHub token "
            f"prefix {VALID_PREFIXES}. Did an extra line get pasted in?"
        )
    # a classic token is 40 chars after the prefix; fine-grained ones are longer
    if len(tok) < 40:
        raise SystemExit(f"The token is only {len(tok)} characters; that looks truncated.")
    return tok


def api(path, token, method="GET", payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"https://api.github.com{path}", data=data, method=method,
                                 headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "casmi-publish",
        **({"Content-Type": "application/json"} if data else {}),
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode("utf-8", "replace")[:300]
    except Exception as e:
        return 0, {}, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--branch", default="main")
    a = ap.parse_args()

    tok = read_token()
    print(f"token          : read from .secrets/github/access_token "
          f"({len(tok)} chars, prefix {tok[:11]!r}) — not printed")

    # --- token identity and repo write permission -----------------------------
    _st, _h, me = api("/user", tok)
    if not isinstance(me, dict) or "login" not in me:
        raise SystemExit(f"token rejected by the API: HTTP {_st} {str(me)[:120]}")
    print(f"authenticated  : {me['login']}")

    _st, _h, repo = api(f"/repos/{OWNER}/{REPO}", tok)
    if not isinstance(repo, dict) or "full_name" not in repo:
        raise SystemExit(f"cannot read the repository: HTTP {_st} {str(repo)[:120]}")
    perms = (repo.get("permissions") or {})
    print(f"repository     : {repo.get('full_name')}  private={repo.get('private')}  "
          f"default_branch={repo.get('default_branch')}  role_push={perms.get('push')}")

    # The role_push field above reports the USER's role on the repository, NOT what this
    # token may do -- a fine-grained token can read a public repo while being unable to
    # write it. Prove write access instead: POST a dangling blob. That needs
    # Contents: read and write, creates no commit and no ref, and is garbage-collected.
    st, _h, body = api(f"/repos/{OWNER}/{REPO}/git/blobs", tok, method="POST",
                       payload={"content": "write-permission probe", "encoding": "utf-8"})
    if st == 201:
        print("write access   : confirmed (a test blob was accepted and discarded)")
    elif st == 409:
        # GitHub answers 409 "Git Repository is empty" for blob creation on a repo with no
        # commits -- which is exactly the first-push case. Nothing to conclude, so proceed.
        print("write access   : not probeable (the repository is still empty) — proceeding")
    elif st in (403, 404):
        raise SystemExit(
            "This token can read the repository but not write to it.\n"
            "Fix it at https://github.com/settings/personal-access-tokens -> open this token ->\n"
            "  Repository permissions -> Contents -> change to 'Read and write' -> Save.\n"
            "The same token then works; there is no need to regenerate it.\n"
            f"(GitHub said: {body})"
        )
    else:
        print(f"write probe    : inconclusive (HTTP {st}) — continuing anyway")

    # --- remote and the two network settings git needs ------------------------
    rc, out, _ = git("remote")
    if "origin" not in out.split():
        git("remote", "add", "origin", REMOTE)
        print("remote         : origin added")
    else:
        git("remote", "set-url", "origin", REMOTE)
        print("remote         : origin set")
    git("config", "http.proxy", PROXY)
    git("config", "https.proxy", PROXY)
    git("config", "http.sslBackend", "openssl")
    print(f"git network    : http.proxy={PROXY}  http.sslBackend=openssl (repo-local)")

    rc, out, err = git("ls-remote", "--heads", "origin")
    if rc != 0:
        raise SystemExit(f"cannot reach GitHub: {err.strip()[:200]}")
    print(f"connectivity   : ok — remote has {len(out.splitlines())} branch(es)")

    if a.check_only:
        print("\ncheck-only: all good, nothing pushed.")
        return 0

    # --- push -----------------------------------------------------------------
    # The token goes in an Authorization header supplied through git's config-from-
    # environment mechanism: nothing lands on the command line, in .git/config or on disk.
    b64 = base64.b64encode(f"x-access-token:{tok}".encode()).decode()
    env = {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.extraHeader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {b64}",
    }
    print(f"\npushing        : {a.branch} -> origin")
    rc, out, err = git("push", "origin", f"{a.branch}:{a.branch}", env=env)
    for line in (err or out).splitlines():
        if line.strip():
            # defensively make sure a token can never be echoed back
            print("   ", line.replace(tok, "<redacted>")[:160])
    if rc != 0:
        raise SystemExit(f"push failed with exit {rc}")
    print(f"\ndone — https://github.com/{OWNER}/{REPO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
