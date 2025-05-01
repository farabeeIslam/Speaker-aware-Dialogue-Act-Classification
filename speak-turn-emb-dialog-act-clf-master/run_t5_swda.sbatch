  GNU nano 2.3.1                                                               File run_t5_swda.sbatch

#!binbash
#SBATCH --job-name=t5_swda
#SBATCH --partition=gpu           # Use GPU A100 partition
#SBATCH --gres=gpu1                   # Request 1 GPU
#SBATCH --time=003000                # Max runtime 40 minutes
#SBATCH --mem=8G
#SBATCH --output=T5_swda_fast_output.txt
#SBATCH --error=T5__swda_fast_error.txt


# Load Anaconda and activate your environment
module load Anaconda32023.03-1
source activate cs5293-3

# Run your script
python run_t5_swda.py

