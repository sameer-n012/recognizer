# are you sure you want to delete all files?

read -p "This will delete all files in the data directory. Are you sure? (y/n) " -n 1 -r
echo    # move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm data/clusters/refined/*
    rm data/clusters/*.json
    rm data/clusters/*.parquet

    rm detections/faces/*
    rm detections/persons/*

    rm embeddings/body/*
    rm embeddings/clip/*
    rm embeddings/face/*
    rm embeddings/fused/*

    rm entities/*.json

    rm -rf frames/*

    rm raw/images/*
    rm raw/videos/*

    echo "All files in the data directory have been deleted."
