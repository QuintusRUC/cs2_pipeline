import subprocess
import sys

def run(script_name: str):
    print(f"\n=== Running {script_name} ===")
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    print(result.stdout)

    if result.returncode != 0:
        print("ERROR output:")
        print(result.stderr)
        raise SystemExit(f"Stopped because {script_name} failed.")

if __name__ == "__main__":
    run("main.py")
    run("price_history_all_daily_30d.py")
    run("item_features.py")
    run("merge_master_daily.py")

    print("\n Finished. The final file is: master_daily_30d.csv")


