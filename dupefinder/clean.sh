# are you sure you want to delete all files?

read -p "This will delete the file index, all logs, and everything under data/ (detections, hashes, embeddings, candidates, duplicate groups — including any manual dashboard edits in group_overrides.parquet). Are you sure? (y/n) " -n 1 -r
echo ""   # move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf data/*

    rm -f cache/file_index.parquet

    rm -f logs/*.log

    echo "Data, file index, and logs have been deleted."
fi
