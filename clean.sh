# are you sure you want to delete all files?

read -p "This will delete all files in the data directory. Are you sure? (y/n) " -n 1 -r
echo ""   # move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm data/clusters/refined/*
    rm data/clusters/*.npy
    rm data/clusters/*.parquet

    # rm data/detections/faces/*
    # rm data/detections/persons/*

    rm data/embeddings/body/*
    rm data/embeddings/clip/*
    rm data/embeddings/face/*
    rm data/embeddings/fused/*

    rm data/entities/*.json

    # rm -rf data/frames/*

    # rm data/raw/images/*
    # rm data/raw/videos/*

    echo "All files in the data directory have been deleted."
fi
