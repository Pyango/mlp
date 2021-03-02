#!/bin/bash

LAST_TAG=`git describe --abbrev=0 --tags`
LAST_TAG_ARRAY=(${LAST_TAG//./ })
SHA_LAST_TAG=`git show-ref -s $LAST_TAG`

echo "Last tag: $LAST_TAG"
echo "Tag array: $LAST_TAG_ARRAY"
echo "Commit: $CI_COMMIT_SHA"
echo "Last tag SHA: $SHA_LAST_TAG"

if [[ $CI_COMMIT_SHA != $SHA_LAST_TAG ]]; then
  if [[ $CI_COMMIT_MESSAGE == *"#major_release"* ]]; then
    echo "major release coming!"
    LAST_TAG_ARRAY[1]=$((${LAST_TAG_ARRAY[1]}+1))
    NEW_TAG=${LAST_TAG_ARRAY[0]}.${LAST_TAG_ARRAY[1]}.0
    echo "release number is $NEW_TAG"
  elif [[ $CI_COMMIT_MESSAGE == *"#public_release"* ]]; then
    echo "public release coming!"
    LAST_TAG_ARRAY[0]=$((${LAST_TAG_ARRAY[0]}+1))
    NEW_TAG=${LAST_TAG_ARRAY[0]}.0.0
    echo "release number is $NEW_TAG"
  else
    echo "minor release coming!"
    LAST_TAG_ARRAY[2]=$((${LAST_TAG_ARRAY[2]}+1))
    NEW_TAG=${LAST_TAG_ARRAY[0]}.${LAST_TAG_ARRAY[1]}.${LAST_TAG_ARRAY[2]}
    echo "release number is $NEW_TAG Project: $CI_PROJECT_ID"
    echo https://gitlab.pyango.ch/api/v4/projects/$CI_PROJECT_ID/releases
  fi
  curl --form "name=Release $NEW_TAG" \
     --form "tag_name=$NEW_TAG" \
     --form "ref=$CI_COMMIT_REF_NAME" \
     --form "description=$CI_COMMIT_MESSAGE" \
     --request POST https://gitlab.pyango.ch/api/v4/projects/$CI_PROJECT_ID/releases?access_token=$CI_JOB_TOKEN
else
  echo "Commit was already tagged. Will not tag and release again."
fi
