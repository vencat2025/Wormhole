import sys
import time
import pexpect

def main():
    print("=== Launching Clean Interactive Codex v0.147.0 Session ===")
    child = pexpect.spawn("codex", dimensions=(35, 120), encoding="utf-8", timeout=60)
    child.logfile = sys.stdout

    # Wait for the EXACT string that unlocks the input prompt
    print("\n[AUTOMATION] Waiting for MCP server boot warning ('not initialized: codex_apps')...")
    child.expect("not initialized: codex_apps", timeout=25)
    
    time.sleep(2)
    print("\n[AUTOMATION] Sending prompt text...")
    child.send("Review the code in this directory.")
    time.sleep(1)
    
    print("\n[AUTOMATION] Submitting prompt with Option+Enter (\\x1b\\r)...")
    child.send("\x1b\r")
    time.sleep(1)
    child.send("\r")

    # Read output stream for 35 seconds
    start_t = time.time()
    while time.time() - start_t < 35:
        try:
            buf = child.read_nonblocking(size=4096, timeout=1)
            sys.stdout.write(buf)
            sys.stdout.flush()
        except pexpect.TIMEOUT:
            pass
        except pexpect.EOF:
            print("\n=== CODEX INTERACTIVE SESSION CLOSED ===")
            break

if __name__ == "__main__":
    main()
