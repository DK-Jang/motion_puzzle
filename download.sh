FILE=$1

if [ $FILE == "pretrained-network" ]; then
    URL=https://www.dropbox.com/s/73o0wu4z1f7y5k0/model_ours.zip?dl=0
    ZIP_FILE=./model_ours.zip
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE
    rm $ZIP_FILE

elif  [ $FILE == "datasets" ]; then
    URL=https://www.dropbox.com/s/91dgn3ktc1gdkrm/datasets.zip?dl=0
    ZIP_FILE=./datasets/datasets.zip
    mkdir -p ./datasets
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./datasets
    rm $ZIP_FILE

else
    echo "Available arguments are pretrained-network, and datasets."
    exit 1

fi