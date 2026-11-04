"""Reject unverified checkout overrides before loading any production secrets."""
import os
import sys


def compatible_checkout(env):
    repository = env.get("CHECKOUT_REPOSITORY", "")
    ref = env.get("CHECKOUT_REF", "")
    return (
        repository in ("", env.get("GITHUB_REPOSITORY"))
        and ref in ("", "main", "refs/heads/main")
    )


if __name__ == "__main__" and not compatible_checkout(os.environ):
    sys.exit("REPOSITORY/REF overrides need review before this upgrade can run.")
