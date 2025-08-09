# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

# prereq:
# 1. Install jq
# 2. Install aws cli

# Add model: aws configure add-model --service-model file://./codegurureviewerv2-2018-05-10.normal.json --service-name codeguru-security
# install brew: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
# install jq using: brew install jq

set -e # exit on first error

# Run script ./run_codeguru_security.sh MyScan upload_folder/zipFile us-east-1, region is optional

scanName="$1"
fileOrFolder="$2"
region="$3"
outputFile="$4"

die() { echo "$*" 1>&2 ; exit 1; }


zipName="/tmp/$(date +%s).zip"

[ "$#" -eq 4 ] || die "4 arguments required. Usage: $0 <scanName> <fileOrFolder> <region> <outputFile>"

echo "$fileOrFolder"

if [ ! -d "$fileOrFolder" ] && [ ! -f "$fileOrFolder" ]; then
        die "file or folder doesn't exist"
fi
if [ -d "$fileOrFolder" ]; then
  zipName="/tmp/$(date +%s).zip"
  zip -r "$zipName" "$fileOrFolder"
else
  zipName=$fileOrFolder
fi

if [[ -z "$region" ]]; then
  region=$(aws configure get region)
fi

if [[ -z "$region" ]]; then
  die "no region provided in script and no default region is present aws configuration"
fi


createuploadcommand="aws codeguru-security create-upload-url --region $region --scan-name=$scanName"
echo -e "Uploading content\n"
echo $createuploadcommand

uploadUrl=$(eval $createuploadcommand)

echo $uploadUrl

### Extracting variables
s3Url=$(echo $uploadUrl | jq '.s3Url')
requestHeaders=$(echo $uploadUrl | jq -r '.requestHeaders| to_entries | map("-H \""+ (.key) + ":" + (.value|tostring) + "\"")| join(" ")' )
codeArtifactId=$(echo $uploadUrl | jq '.codeArtifactId')

uploadContentCommand="curl -X PUT -T $zipName -H \"Content-Type: application/zip\" $requestHeaders $s3Url"


echo "Uploading content by running following command.\n"
echo $uploadContentCommand

eval $uploadContentCommand

createScanCommand="aws codeguru-security create-scan --region $region --scan-name=$scanName --resource-id '{\"codeArtifactId\": $codeArtifactId}'"

echo -e "creating a scan \n"

echo $createScanCommand

scan=$(eval $createScanCommand)


runId=$(echo $scan | jq '.runId')


echo $scan

scanState="InProgress"
getCommand="aws codeguru-security get-scan --region $region --scan-name=$scanName --run-id=$runId"

inprogress="InProgress"
while [ $scanState = $inprogress ]
do
    echo "Running Get to check if status is completed"

    echo $getCommand

    getscanOut=$(eval $getCommand)
    scanState=$(echo $getscanOut | jq '.scanState' | tr -d '"')

    echo "Current scanState: $scanState, expected $inprogress"

    sleep 10
done

getFindingsCommand="aws codeguru-security get-findings --region $region --scan-name=$scanName --output json | tee $outputFile"

echo $getFindingsCommand

eval $getFindingsCommand

echo "Findings written to $outputFile"
