##############################################################
# Remember checking laptop_swing.json for number of generations
##############################################################
cd weightagnostic_original/prettyNeatWann

##############################################################
# Run training
##############################################################
# Train normal model
python3.9 wann_train.py -p p/laptop_swing.json -n 8 -o "normal" --normal_pendulum=1
# Train fewshot model
python3.9 wann_train.py -p p/laptop_swing.json -n 8 -o "fewshot" --fewshot=true --min_pendulum=0.5 --max_pendulum=1.5

##############################################################
# Set test parameters
##############################################################
pendulum_weights=`seq 0.5 0.05 1.5`
n_repetitions=10
n_weigths=100
echo "${pendulum_weights}" > ../../results/result_weigths.txt

##############################################################
# Run tests
##############################################################
# Test normal model
for i in ${pendulum_weights}; do python3.9 wann_test.py -p p/laptop_swing.json -i log/normal_best.out --nReps="${n_repetitions}" --view False --nVals="${n_weigths}" -o "normal_${i}_" --normal_pendulum="${i}"; done
# Collect results of normal model
for i in ${pendulum_weights}; do sort -gr "normal_${i}_reward.out" | head -n 1 >> ../../results/normal_results.txt; done
# Test fewshot model
for i in ${pendulum_weights}; do python3.9 wann_test.py -p p/laptop_swing.json -i log/fewshot_best.out --nReps="${n_repetitions}" --view False --nVals="${n_weigths}" -o "fewshot_${i}_" --normal_pendulum="${i}"; done
# Collect results of fewshot model
for i in ${pendulum_weights}; do sort -gr "fewshot_${i}_reward.out" | head -n 1 >> ../../results/fewshot_results.txt; done
