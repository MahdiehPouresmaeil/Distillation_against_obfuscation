#!/bin/bash

methods=(  "DICTION"  ) # values: ("DICTION" "DEEPSIGNS" "UCHIDA" "RES_ENCRYPT" "PASSPORT" "HUFUNET" "STDM" "RIGA" "GREEDY" "LAPLACIAN" )


models=("CNN") # values:# ("MLP" "CNN" "RESNET18" "VGG16")
operations=( "TRAIN"  "WATERMARKING" ) # values: ("TRAIN" "WATERMARKING" "TRAIN-WATERMARKING" "PRUNING" "OVERWRITING" "FINE_TUNING" "DUMMY_NEURONS" "DISTILLATION""AMBIGUITY")


for method in "${methods[@]}"; do
    for model in "${models[@]}"; do
        for operation in "${operations[@]}"; do
            echo -e "\nRunning $method with $model and $operation"
            echo "START TIME: $(date '+%Y-%m-%d %H:%M:%S')"

            # Determine the output directory
            if [ "$operation" == "TRAIN" ]; then
                output_dir="outs/$operation"
            else
                output_dir="outs/$operation/$method"
            fi

            # Createt the output directory if it does not exist
            mkdir -p "$output_dir"

            # Execute the python script and output the results to a file
            python test_case.py --method "$method" --model "$model" --operation "$operation" | tee -a "$output_dir/$model.txt"
        done
    done
done
echo -e "\n========================================"
echo "All operations completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo "Operations completed."
