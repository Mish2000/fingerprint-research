from pathlib import Path
plain_dir = Path(r"C:\Users\sirak\Downloads\fingerprint collected data\NIST\sd300b\images\1000\png\plain")
roll_dir  = Path(r"C:\Users\sirak\Downloads\fingerprint collected data\NIST\sd300b\images\1000\png\roll")

print("plain sample:")
for p in list(plain_dir.glob("*.png"))[:10]:
    print(" ", p.name)

print("\nroll sample:")
for p in list(roll_dir.glob("*.png"))[:10]:
    print(" ", p.name)
