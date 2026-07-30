# THUCNews Data

This directory contains the reproducible corpus acquisition layer. It downloads
only [`SirlyDreamer/THUCNews`](https://huggingface.co/datasets/SirlyDreamer/THUCNews)
from Hugging Face and converts article titles and bodies into one sentence per
line.

```bash
make -C data status
make -C data download
make -C data process
```

Downloads land in `data/downloads/`; the converted corpus is
`data/derived/THUCNews.sentences.txt`. Both are generated and ignored by git.
