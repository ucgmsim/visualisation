#!/usr/bin/env python

import pandas as pd
import numpy as np
import glob
import os
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute logarithmic mean/median of IM values across realisations of a given fault"
    )
    parser.add_argument(
        "csv_path",
        help="path to directory containing CSV files to recursively search",
        type=os.path.abspath,
    )  # if set too deep, search will be slow.
    parser.add_argument("fault_name", help="name of the fault to process")
    parser.add_argument(
        "--median",
        dest="stat_fn",
        action="store_const",
        const=np.median,
        default=np.mean,
        help="find median (default: mean)",
    )
    parser.add_argument(
        "--im",
        dest="im_types",
        nargs="+",
        type=str,
        help="list of IM types. all chosen if unspecified, all pSA if 'pSA' given",
    )  # eg. --im PGV PGA
    parser.add_argument(
        "--output",
        dest="output_dir",
        help="path for the output CSV file to be saved",
        default=os.path.curdir,
        type=os.path.abspath,
    )

    args = parser.parse_args()

    csv_path = args.csv_path

    fault_name = args.fault_name
    stat_fn = args.stat_fn  # default: np.mean
    im_types = args.im_types
    output_dir = args.output_dir

    if os.path.exists(csv_path) and os.path.isdir(csv_path):
        print("Checked: CSV search directory {}".format(csv_path))
    else:
        print("Error: invalid path : {}".format(csv_path))
        sys.exit(0)

    im_csv_paths = glob.glob(
        os.path.join(csv_path, "**", f"{fault_name}_REL*.csv"), recursive=True
    )
    im_csv_paths.sort()

    if len(im_csv_paths) > 0:
        print("Checked: IM csv files located")
    else:
        print("Error: no IM csv files found")
        sys.exit(0)

    print(im_csv_paths)

    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        print("Checked: Output directory {}".format(output_dir))
    else:
        os.mkdir(output_dir)
        print("Created: Output directory {}".format(output_dir))

    rel_im_dfs = []
    for c in im_csv_paths:
        df = pd.read_csv(c, index_col=0)
        rel_im_dfs.append(df)

    stations = list(rel_im_dfs[0].index)
    stations.sort()

    # check IM types. If unspecified, use all IM_types
    wrong_im_count = 0
    all_psa = False
    if im_types is None:
        im_types = list(rel_im_dfs[0].columns)
        im_types.remove("component")
    else:
        for im_type in im_types:
            if im_type in rel_im_dfs[0].columns:
                pass
            elif im_type == "pSA":
                all_psa = True
            else:
                print("Error: Unknown IM type {}".format(im_type))
                wrong_im_count += 1
        if wrong_im_count > 0:
            sys.exit("Error: Fix IM types")
    if all_psa:
        for avail in rel_im_dfs[0].columns:
            if avail.startswith("pSA"):
                im_types.append(avail)
        im_types.remove("pSA")

    print(
        "Summarising IM values at {} stations from {} realisations for IM types {}".format(
            len(stations), len(rel_im_dfs), im_types
        )
    )
    df_dict = {"station": stations, "component": ["geom"] * len(stations)}
    for im_type in im_types:
        print("...{}".format(im_type))
        im_val_concat = pd.concat(
            [np.log(rel_im_dfs[i][im_type]) for i in range(len(rel_im_dfs))]
        )

        log_mean_im = [np.exp(stat_fn(im_val_concat[k])) for k in stations]
        log_stdev_im = [np.std(im_val_concat[k]) for k in stations]
        modified_im_name = (
            im_type if im_type[0] != "p" else f"p{im_type[1:].replace('p', '.')}"
        )
        df_dict[modified_im_name] = log_mean_im
        df_dict[f"{modified_im_name}_sigma"] = log_stdev_im

    log_mean_ims_df = pd.DataFrame(df_dict)
    output_file = os.path.join(output_dir, f"{fault_name}_log_{stat_fn.__name__}.csv")

    log_mean_ims_df.set_index("station").to_csv(output_file)
    print("Completed...Written {}".format(output_file))
