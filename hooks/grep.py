import os, subprocess, sys

grep_found = False
is_windows = sys.platform == "win32"
basename = os.path.join(os.path.dirname(__file__).removesuffix("hooks").rstrip(os.sep), "neurocaps")
files = [
    os.path.join(basename, "analysis", "cap.py"),
    os.path.join(basename, "extraction", "timeseriesextractor.py"),
]

for file in files:
    if not is_windows:
        # Pattern used two negative look behinds to ignore "self" preceded by a backtick or del + whitespace
        # then a negative look ahead for "self" that is not followed by underscore, followed by word.
        cmd = f"grep -P '(?<!`)(?<!del )self\.(?!_)(?=\w+)' {file}"
    else:
        cmd = [
            "powershell",
            "-Command",
            f"Select-String -Path {file} -Pattern '(?<!`)(?<!del )self\.(?!_)(?=\w+)'",
        ]

    # Get std output
    output = subprocess.run(cmd, shell=not is_windows, capture_output=True, text=True)
    if output.stdout:
        grep_found = True
        sys.stdout.write(output.stdout)

# Non-zero exit if there is output
sys.exit(1 if grep_found else 0)
