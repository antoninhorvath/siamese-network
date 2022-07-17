import subprocess

def run(cmd):
    completed = subprocess.run(["powershell", "-Command", cmd], capture_output=True)
    return completed

if __name__ == '__main__':
    id = "1hTJwFwHsVYKH3BnWfOMtCyD0yWCNUyGx"
    name = "dataset.zip"
    hello_command = "Write-Host 'Hello'"
    hello_info = run(hello_command)
    if hello_info.returncode != 0:
        print("An error occured: %s", hello_info.stderr)
    else:
        print("Hello command executed successfully")
