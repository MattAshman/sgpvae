# Run Japanese weather experiment.
main="experiments/japan.py"

# Standard kwargs.
epochs=26

for es in 1 0
do
for likelihood in "nn" "linear"
do
for pn in "factornet"
do
for i in 1 2 3 4 5 6 7
do
python "$main" --model vae --pinference_net $pn --likelihood $likelihood --epochs $epochs --save True --elbo_subset $es --results_dir "./_results/japan/vae/"
done
done
done
done
