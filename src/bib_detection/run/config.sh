OBJECT_DETECTION_PATH="$HOME/models/object_detection"

# We want to ensure that this is run from the root of the project dir
if [ "${PWD##*/}" != "bibo" ]; then
  echo "You need to run this from the project root dir"
  exit 1
fi
