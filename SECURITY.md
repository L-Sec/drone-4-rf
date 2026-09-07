# Security policy

## Supported versions

Security fixes are applied to the latest release and the `main` branch.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not
include exploit details, RF recordings, event databases, local configuration,
device serial numbers, addresses, or other sensitive material in a public issue.

If private vulnerability reporting is not yet available, open a public issue
containing only a request for a private contact channel. Include no technical
details until that channel is established.

## Sensitive RF data

The files under `data/`, `config/local.yaml`, RF/IQ captures, databases, model
artifacts, and generated build outputs are intentionally excluded from Git. They
can reveal frequency use, activity timing, equipment fingerprints, or details
about the machine and site where Drone 4-RF was run.

Before sharing any exported data, confirm that you are authorized to collect and
distribute it and inspect the export for site, contributor, and device details.
