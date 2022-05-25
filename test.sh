export NAME=human && envsubst < model.yaml | kubectl create -f - 
export NAME=mouse && envsubst < model.yaml | kubectl create -f - 
export NAME=pancreas && envsubst < model.yaml | kubectl create -f - 