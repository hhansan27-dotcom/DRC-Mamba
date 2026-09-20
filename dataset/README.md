# Dataset layout

Place each dataset under this directory using the following structure:

```text
dataset/
├── NUAA-SIRST/
│   ├── images/
│   ├── masks/
│   └── 80_20/
│       ├── train.txt
│       └── test.txt
├── IRSTD-1K/
│   ├── images/
│   ├── masks/
│   └── 80_20/
│       ├── train.txt
│       └── test.txt
└── NUDT-SIRST/
    ├── images/
    ├── masks/
    └── 50_50/
        ├── train.txt
        └── test.txt
```

Each split text file contains one image ID per line, without the file suffix. By default the code uses `.png`; use `--suffix` if your copy uses another extension.

The repository intentionally does not redistribute dataset images or annotations. Please obtain the datasets from their official project pages and follow their licenses/terms.
