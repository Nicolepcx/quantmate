import subprocess


def install_packages():
    packages = ["transformers == 4.26.1", "datasets == 2.10.1", "evaluate==0.4.0", "pyarrow==9.0.0",
                "sentencepiece", "yfinance==0.2.37", "pandas_datareader==0.10.0", "pandas_market_calendars==4.4.0", "huggingface_hub==0.21.4"]
    check = u'\u2705'
    print("\033[1mInstalling requirements...\n\033[0m")
    for package in packages:
        process_scatter = subprocess.run(
            ["pip", "install", package],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process_scatter.returncode != 0:
            print(f"Installation of {package} failed with error:\n{process_scatter.stderr.decode('utf-8')}")
        else:
            print(f"{check} {package} installation completed successfully!\n")


