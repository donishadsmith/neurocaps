import os, subprocess, sys

from typing import Union

IS_WINDOWS: bool = sys.platform == "win32"
BASENAME: str = os.path.join(os.path.dirname(__file__).removesuffix("hooks").rstrip(os.sep), "neurocaps")
FILES : list[str] = [
    os.path.join(BASENAME, "analysis", "cap", "cap.py"),
    os.path.join(BASENAME, "extraction", "timeseries_extractor.py"),
]

def get_cmd(filename: str) -> Union[str, list[str]]:
    if not IS_WINDOWS:
        # Pattern used two negative look behinds to ignore "self" preceded by a backtick or del + whitespace
        # then a negative look ahead for "self" that is not followed by underscore, followed by word.
        cmd = fr"grep -P '(?<!`)(?<!del )self\.(?!_)(?!return_cap_labels\b)(?=\w+)' {filename}"
    else:
        cmd = [
            "powershell",
            "-Command",
            fr"Select-String -Path {filename} -Pattern '(?<!`)(?<!del )self\.(?!_)(?!return_cap_labels\b)(?=\w+)'",
        ]

    return cmd

def get_stdout(cmd: Union[str, list[str]]) -> str:
    output = subprocess.run(cmd, shell=not IS_WINDOWS, capture_output=True, text=True)

    return output.stdout

def main() -> None:
    grep_output = False

    for file in FILES:
        cmd = get_cmd(file)
        if standard_output := get_stdout(cmd):
            grep_output = True
            sys.stdout.write(standard_output)

    sys.exit(1 if grep_output else 0)

if __name__ == "__main__":
    main()
