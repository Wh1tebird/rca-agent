# Security policy

## Scope

The public repository contains the runtime and synthetic validation code. RCA100 observation
data, provider credentials, model transcripts, and private research artifacts are external and
must not be committed.

The local FastAPI console is a development host. Run it on loopback, keep provider credentials in
the host environment, and do not expose it directly to an untrusted network without adding
authentication, TLS, and an appropriately isolated run directory.

## Reporting

Do not open a public issue for a credential, private dataset, or other sensitive disclosure. Revoke
the credential immediately, preserve only the minimum diagnostic evidence, and contact the project
maintainer through the private channel configured for the release.
