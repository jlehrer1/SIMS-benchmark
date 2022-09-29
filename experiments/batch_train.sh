files=("primary_T.h5ad" "allen/human.h5ad")
names=("bhaduri-1-gpu-smaller-latent-batchnorm-big-gpu" "allen-1-gpu-smaller-latent-batchnorm-big-gpu")

for i in "${!files[@]}"; do
    export NAME=${names[i]} && export FILE=${files[i]} export BATCH=4 && envsubst < yaml/autoencoder.yaml | kubectl create -f -
done