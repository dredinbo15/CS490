#!/usr/bin/env bash
# Downloads the YAMNet TFLite model and its class-name list into pi_service/models/.
# If a download fails (links move), get them by hand:
#   model:     https://www.kaggle.com/models/google/yamnet  -> TFLite -> download, rename to yamnet.tflite
#   class map: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

curl -fL -o models/yamnet.tflite \
  "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"
curl -fL -o models/yamnet_class_map.csv \
  "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

echo "Downloaded:"
ls -lh models/
echo "Glass-related classes in the class map:"
grep -inE "glass|shatter|break|clink" models/yamnet_class_map.csv || true
