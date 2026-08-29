

import ctypes
import time
import sys

# Define Windows API constants
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

def prevent_sleep():
    if sys.platform != 'win32':
        print("This script is designed for Windows platforms only.")
        return False
        
    print("Setting system execution state to prevent sleep...")
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    # Keeps the system and screen awake.
    result = ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )
    return result != 0

def restore_sleep():
    if sys.platform == 'win32':
        print("Restoring default sleep settings...")
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

def main():
    if sys.platform != 'win32':
        print("This script is only supported on Windows.")
        return

    success = prevent_sleep()
    if not success:
        print("Failed to set execution state.")
        return

    print("\n" + "="*50)
    print("  LAPTOP WAKE-LOCK ACTIVE")
    print("  The screen and system will remain awake.")
    print("  Press Ctrl+C to stop and restore sleep settings.")
    print("="*50 + "\n")

    try:
        while True:
            # Wake locks need to be renewed periodically or just kept running
            time.sleep(60)
            # Optional: Move mouse cursor slightly or just keep system state continuous
    except KeyboardInterrupt:
        print("\nStopping wake lock...")
    finally:
        restore_sleep()
        print("Exit.")

if __name__ == "__main__":
    main()
