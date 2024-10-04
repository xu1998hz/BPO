from safetensors import safe_open
from safetensors.torch import save_file
import click
import os

@click.command()
@click.option('-weight_prefix', type=str)
@click.option('-ckpt', type=str)
def main(weight_prefix, ckpt):
    num_weights=8
    # averge model weights into a unified parametric space
    for partition in ["model-00001-of-00002", "model-00002-of-00002"]:
        final_tensors = {}
        for i in range(num_weights):
            f = safe_open(f"/share/edc/home/wendaxu/dpo/{i}_dpo_{weight_prefix}/{ckpt}/{partition}.safetensors", framework="pt", device='cpu')
            if len(final_tensors) == 0:
                for k in f.keys():
                    final_tensors[k] = f.get_tensor(k)
            else:
                for k in final_tensors.keys():
                    final_tensors[k] += f.get_tensor(k)
        for k in final_tensors.keys():
            final_tensors[k]/=num_weights

        if not os.path.exists(f"/share/edc/home/wendaxu/dpo/all_dpo_{weight_prefix}"):
            os.mkdir(f"/share/edc/home/wendaxu/dpo/all_dpo_{weight_prefix}") 
        save_file(final_tensors, f"/share/edc/home/wendaxu/dpo/all_dpo_{weight_prefix}/{partition}.safetensors", metadata={'format': 'pt'})
        print(f"/share/edc/home/wendaxu/dpo/all_dpo_{weight_prefix}/{partition}.safetensors is saved!")

if __name__ == "__main__":
    main()  