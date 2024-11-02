# Variables
model_name="10-23-2024-09:18:50"
pdb="7za3"
config="trial_7.cfg"

# Paths
model="models/${model_name}.keras"
base_dir="dataset/${pdb}"
source="${base_dir}/source.map"
mask="${base_dir}/target.map"
output_dir="tests/${model_name}/${pdb}"
config_path="configurations/${config}"

mkdir -p ${output_dir}

python glycan_prediction_model/inference/infer.py -m ${model} -i ${source} --mask ${mask} -o ${output_dir} -c ${config_path}

output_map="${output_dir}/output.map"
reformatted_output_map="${output_dir}/output_reformatted.map"

python map_convert.py -i ${output_map} -o ${reformatted_output_map}