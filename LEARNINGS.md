## Effects of Chunking on RAG Performance

### For a single file
```bash
python src/evaluation/experiment_chunking.py \
  --input data/raw/iliad.txt \
  --queries configs/queries_myths_iliad.yaml \
  --out data/experiments/chunking/iliad_minilm/ \
  --embed_model sentence-transformers/all-MiniLM-L6-v2 \
  --sizes 200,300,500,800 \
  --overlaps 0,50,100 \
  --ks 1,2,3,5,10
```
```bash
python src/evaluation/experiment_chunking.py \
  --input data/raw/iliad.txt \
  --queries configs/queries_myths_beowolf.yaml \
  --out data/experiments/chunking/beowolf_minilm/ \
  --embed_model sentence-transformers/all-MiniLM-L6-v2 \
  --sizes 200,300,500,800 \
  --overlaps 0,50,100 \
  --ks 1,2,3,5,10
```

### Over multiple files
```bash
python src/evaluation/experiment_chunking.py \
  --input data/raw/ \
  --queries configs/queries_myths.yaml \
  --out data/experiments/chunking/all_minilm/
```


### All books
```bash
python src/evaluation/experiment_chunking.py \
  --input data/raw/ \
  --queries configs/queries_myths.yaml
  --ks 1,2,3,5,10
```

### Embedding Swap
```bash
python src/evaluation/experiment_chunking.py \
  --input data/raw/iliad.txt \
  --queries configs/queries_myths_iliad.yaml \
  --embed_model sentence-transformers/all-mpnet-base-v2
```