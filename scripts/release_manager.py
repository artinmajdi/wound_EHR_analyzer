#!/usr/bin/env python3

import re
import subprocess
import sys


def get_latest_version():
    """
    Retrieves the latest git tag that follows semantic versioning (vX.Y.Z).
    If no tag exists, returns 'v0.0.0'.
    """
    try:
        result = subprocess.run(["git", "tag"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        tags = result.stdout.strip().splitlines()
        version_tags = []
        regex = re.compile(r'^v(\d+)\.(\d+)\.(\d+)$')
        for tag in tags:
            m = regex.match(tag)
            if m:
                major, minor, patch = map(int, m.groups())
                version_tags.append((major, minor, patch, tag))
        if not version_tags:
            return "v0.0.0"
        # Sort and return the highest version
        version_tags.sort(key=lambda x: (x[0], x[1], x[2]))
        return version_tags[-1][3]
    except subprocess.CalledProcessError:
        return "v0.0.0"


def bump_version(latest_version, bump_type):
    """
    Bumps the version based on the provided bump type: 'major', 'minor', or 'patch'.
    """
    regex = re.compile(r'^v(\d+)\.(\d+)\.(\d+)$')
    m = regex.match(latest_version)
    if not m:
        print("Latest version doesn't follow semantic versioning. Aborting.")
        sys.exit(1)
    major, minor, patch = map(int, m.groups())
    if bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    elif bump_type == 'patch':
        patch += 1
    else:
        print("Invalid bump type specified.")
        sys.exit(1)
    return f"v{major}.{minor}.{patch}"


def create_new_release(version):
    """
    Creates a new annotated git tag for the release and pushes it to origin.
    """
    tagline = f"Release {version}: New release"
    try:
        subprocess.run(["git", "tag", "-a", version, "-m", tagline], check=True)
        subprocess.run(["git", "push", "origin", version], check=True)
        print(f"Successfully created and pushed tag {version}")
    except subprocess.CalledProcessError as e:
        print("Error creating or pushing the tag:", e)
        sys.exit(1)


def main():
    latest = get_latest_version()
    print(f"Latest version is: {latest}")
    print("What type of release bump would you like to create?")
    print("Options: major, minor, patch, or custom")
    bump = input("Enter your choice: ").strip().lower()
    if bump == "custom":
        custom_version = input("Enter the custom version tag (e.g., v2.0.0): ").strip()
        new_version = custom_version
    elif bump in ["major", "minor", "patch"]:
        new_version = bump_version(latest, bump)
    else:
        print("No valid bump type selected. Exiting.")
        sys.exit(0)

    confirm = input(f"Do you want to create a new release {new_version}? (y/N): ").strip().lower()
    if confirm in ["y", "yes"]:
        create_new_release(new_version)
    else:
        print("No release created.")


if __name__ == '__main__':
    main()
