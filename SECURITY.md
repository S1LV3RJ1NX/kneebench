# Security policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub security advisories](https://github.com/S1LV3RJ1NX/kneebench/security/advisories/new).
Do not open a public issue. We aim to acknowledge reports within 3 working days.

## How kneebench handles API keys

- Keys come from the `KNEEBENCH_API_KEY` environment variable or the web UI session.
  kneebench never writes them to its database, config files, reports or exports.
- kneebench passes the key to AIPerf only through the child process environment.
  The generated AIPerf config holds a `${KNEEBENCH_AIPERF_API_KEY}` placeholder, so
  the key never appears in a command line (visible to other users via `ps`) or in a file.
- AIPerf redacts the key in its JSON exports. Because its error messages can echo
  config values, kneebench also scrubs every log and text artifact after each run.
- A test runs the bundled AIPerf with a sentinel key and fails if the key appears in
  any artifact.

Residual risk: on the machine running kneebench, other processes running as the
**same user** can read a process's environment. Run kneebench under a user you trust.

## Scope

kneebench only sends benchmark traffic to the endpoint you give it. It never runs
commands you paste, and it never starts or manages servers.
