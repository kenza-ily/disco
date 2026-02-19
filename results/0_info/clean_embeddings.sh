#!/bin/bash
# Clean old embedding versions, keep only newest

mkdir -p ../../archive/old_embeddings_20260219

cd ../3_embeddings

# For each dataset directory
for dataset_dir in */; do
    if [ -d "$dataset_dir" ]; then
        echo "Processing $dataset_dir"
        cd "$dataset_dir"

        # Find all embedding files and group by base name (without timestamp)
        for base_name in $(ls *.npy 2>/dev/null | sed 's/_[0-9]\{8\}_[0-9]\{6\}\.npy$//' | sort -u); do
            # List all versions of this embedding, sorted by timestamp (newest first)
            files=$(ls -t "${base_name}"_*.npy 2>/dev/null)

            if [ $(echo "$files" | wc -l) -gt 1 ]; then
                echo "  Found multiple versions of ${base_name}, keeping newest"
                # Skip first (newest), move rest to archive
                echo "$files" | tail -n +2 | while read old_file; do
                    echo "    Archiving: $old_file"
                    mv "$old_file" "../../../archive/old_embeddings_20260219/"
                done
            fi
        done

        cd ..
    fi
done

echo "✓ Embedding cleanup complete. Old versions moved to archive/old_embeddings_20260219/"
