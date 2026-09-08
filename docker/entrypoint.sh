#!/bin/sh
set -e

DATASET_ROOT="${INTERIM_DIR:-/app/data/interim}/ai_assisted_spam"

if [ "${AUTO_PREPARE_DATASET:-1}" = "1" ] && [ ! -d "$DATASET_ROOT" ] && [ -f /app/data/datasets/ai_assisted_spam_dataset.zip ]; then
    email-ai-detector prepare-data
fi

exec email-ai-detector "$@"
