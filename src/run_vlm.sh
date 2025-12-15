#!/bin/bash

: <<'EOF'

EOF

k_samples=(5 10 30 50)  # 100 

v_models=(vit convnext dinov2)  #  convnext dinov2
t_models=(bert allroberta) #  allroberta

b_data="nocaps"    # "nocaps"  "coco" 
q_data="nocaps"    # "nocaps"  "coco"

base_command="python ./src/run_vlm.py --gpu 0"
timestamp=$(date +"%Y%m%d_%H%M%S")

distance="Cosine"           # "Cosine"   "Euclidean"

for v_model in "${v_models[@]}"; do 
    for t_model in "${t_models[@]}"; do
        for ks in "${k_samples[@]}"; do
            output_file="./results/B_${b_data}_Q_${q_data}_V_${v_model}_T_${t_model}_D_${distance}_${timestamp}.log"
            full_command="$base_command --kkk ${ks} --dist ${distance} \
            --vid_model ${v_model} --text_model ${t_model} \
            --base_data ${b_data}  --query_data ${q_data}   >> $output_file"
            echo "Now, running: $full_command"
            eval $full_command
            echo "Test resuts saved to $output_file"
        done
    done
done