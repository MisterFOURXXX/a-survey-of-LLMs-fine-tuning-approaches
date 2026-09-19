from pathlib import Path
import subprocess

def download_stacksample(out_dir: str = "data/raw") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "stackoverflow/stacksample", "-p", str(out)],
        check=True,
    )
    subprocess.run(
        ["unzip", "-o", str(out / "stacksample.zip"), "-d", str(out / "stacksample")],
        check=True,
    )