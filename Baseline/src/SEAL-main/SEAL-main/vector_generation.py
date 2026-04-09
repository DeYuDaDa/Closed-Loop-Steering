import torch
import os
import argparse

def load_data(data_dir, prefixs, layer_num=29, max_examples=None):
    data_paths = [os.path.join(data_dir, f"hidden_{p}", "hidden.pt") for p in prefixs]
    switch = [[] for _ in range(layer_num)]
    check = [[] for _ in range(layer_num)]
    other = [[] for _ in range(layer_num)]
    for i, data_path in enumerate(data_paths):
        data = torch.load(data_path, weights_only=False)

        for l in range(layer_num):
            layer_data = data[l]
            for k in layer_data:
                if max_examples is not None and max_examples > 0 and k >= max_examples:
                    continue
                h = layer_data[k]["step"]
                check_index = layer_data[k]["check_index"]
                switch_index = layer_data[k]["switch_index"]
                check[l].append(h[check_index])
                switch[l].append(h[switch_index])
                all_indices = torch.arange(h.shape[0])
                mask = ~(torch.isin(all_indices, check_index) | torch.isin(all_indices, switch_index))
                other[l].append(h[mask])
    for l in range(layer_num):
        check[l] = torch.cat(check[l], dim=0)
        switch[l] = torch.cat(switch[l], dim=0)
        other[l] = torch.cat(other[l], dim=0)
    check = torch.stack(check, dim=0)
    switch = torch.stack(switch, dim=0)
    other = torch.stack(other, dim=0)
    return check, switch, other


def generate_vector_switch_check(data_dir, prefixs, layers, save_prefix, overwrite=False):
    if isinstance(layers, int):
        layers = [layers]
    max_layer = max(layers)
    check, switch, other = load_data(data_dir=data_dir, prefixs=prefixs, layer_num=max_layer+1)
    save_dir = os.path.join(data_dir, f"vector_{save_prefix}")
    print(f"save_dir: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    for layer in layers:
        layer_check = check[layer]
        layer_switch = switch[layer]
        layer_other = other[layer]
        steer_vec = torch.cat([layer_check, layer_switch], dim=0).mean(dim=0) - layer_other.mean(dim=0)
        save_path = os.path.join(save_dir, f"layer_{layer}_transition_reflection_steervec.pt")
        if not os.path.exists(save_path) or overwrite:
            torch.save(steer_vec, save_path)
        else:
            print(f"{save_path} already exists")
        print(f"layer {layer} done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--prefixs", type=str, nargs="+", default=["correct_0_500", "incorrect_0_500"])
    parser.add_argument("--layers", type=int, nargs="+", default=[20])
    parser.add_argument("--save_prefix", type=str, default="500_500")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    generate_vector_switch_check(
        data_dir=args.data_dir,
        prefixs=args.prefixs,
        layers=args.layers,
        save_prefix=args.save_prefix,
        overwrite=args.overwrite
    )