import sys
import time
import pexpect

def test_codex_interactive():
    print("Launching codex interactive session...")
    child = pexpect.spawn("codex", dimensions=(35, 120), encoding="utf-8", timeout=30)
    child.logfile = sys.stdout

    time.sleep(3)
    # If update prompt is visible, send option 2 (Skip)
    buffer_text = child.before or ""
    if "Update available" in buffer_text or "runs `brew upgrade" in buffer_text:
        print("\n[INFO] Detected update menu. Sending option 2 (Skip)...")
        child.sendline("2")
        time.sleep(2)

    print("\nSending prompt: Review the code in this directory.")
    child.sendline("Review the code in this directory.")
    
    start_t = time.time()
    while time.time() - start_t < 25:
        try:
            line = child.read_nonblocking(size=2048, timeout=1)
            sys.stdout.write(line)
            sys.stdout.flush()
        except pexpect.TIMEOUT:
            pass
        except pexpect.EOF:
            print("\n--- CODEX SESSION EXITED ---")
            break

if __name__ == "__main__":
    test_codex_interactive()
