import pickle
from itertools import chain
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
import pandas as pd
import requests
import torch
from dagshub.streaming import DagsHubFilesystem
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset
from tqdm import tqdm
import os
from urllib.parse import urlparse

from data.transforms.unpack import UnpackBatch

from .encoded import EncodedDataset

class LAIONAesthetics(TorchDataset):
    def __init__(
        self,
        transform=None,
        dataset_root="./notebooks/dataset/LAION/laion-600k-aesthetic-6.5plus-768",
        silent_timeout=False,
        subset_size=None,
        seed=42,
        extensions=(".jpg", ".jpeg", ".png"),
    ):
        """
        Local-only version of LAIONAesthetics.

        Args:
            transform: A transform to be applied to PILs when __getitem__ is called.
            dataset_root: Directory that directly contains the image files.
            silent_timeout: When True, if loading an image fails, return a black image
                instead of raising.
            subset_size: If not None, randomly sample this many images (without replacement).
            seed: Seed for sampling the subset.
            extensions: Accepted image file extensions.
        """
        self.transform = transform
        self.dataset_root = Path(dataset_root)
        self.silent_timeout = silent_timeout

        # --------------------------------------------------------
        # 1) Collect all image paths under dataset_root
        # --------------------------------------------------------
        all_imgs = []
        for ext in extensions:
            all_imgs.extend(self.dataset_root.glob(f"*{ext}"))

        all_imgs = sorted(all_imgs)  # For deterministic ordering

        if len(all_imgs) == 0:
            raise RuntimeError(f"No images found in {self.dataset_root} with extensions {extensions}")

        # --------------------------------------------------------
        # 2) Optional subset sampling
        # --------------------------------------------------------
        if subset_size is not None and subset_size < len(all_imgs):
            rng = np.random.default_rng(seed)
            idxs = rng.choice(len(all_imgs), size=subset_size, replace=False)
            idxs = np.sort(idxs)
            all_imgs = [all_imgs[i] for i in idxs]

        self.img_paths = all_imgs                 # list[Path]
        self.img_cache = {}                       # path_str -> PIL.Image
        self.captions = [p.name for p in all_imgs]  # use filename as "caption"

        # Optional metadata, if you use it anywhere
        self.metadata = pd.DataFrame(
            {
                "image_path": [str(p) for p in self.img_paths],
                "caption": self.captions,
            }
        )

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]  # Path object
        key = str(img_path)

        if key in self.img_cache:
            img = self.img_cache[key]
        else:
            try:
                img = Image.open(img_path)
            except (FileNotFoundError, OSError) as e:
                if self.silent_timeout:
                    # Just return a blank image
                    img = Image.new("RGB", (512, 512))
                else:
                    raise RuntimeError(f"Error loading image {img_path}: {type(e).__name__}: {e}")
            img = img.convert("RGB")
            self.img_cache[key] = img

        if self.transform is not None:
            img = self.transform(img)

        caption = self.captions[idx]
        return img, caption

    def __len__(self):
        return len(self.img_paths)

    
# class LAIONAesthetics(TorchDataset):
#     def __init__(
#         self,
#         transform=None,
#         dataset_root="./outputs/laion",
#         silent_timeout=False,
#         subset_size=None,
#         seed=42,
#     ):
#         """
#         Args:
#             transform: A transform to be applied to PILs when __getitem__ is called.
#             dataset_root: Location where downloaded data will be stored on disk.
#             silent_timeout: When True, if loading an image throws a connection error, catch
#                 and return a black image.
#             subset_size: If not None, creates a subset of LAION Aesthetics with subset_size
#                 randomly sampled images (without replacement).
#             seed: Seed for sampling the subset.
#         """
#         self.transform = transform
#         self.dataset_root = Path(dataset_root)
#         self.silent_timeout = silent_timeout

#         # Setup data streaming from DagsHub
#         try:
#             self.fs = DagsHubFilesystem(
#                 project_root=self.dataset_root,
#                 repo_url="https://dagshub.com/DagsHub-Datasets/LAION-Aesthetics-V2-6.5plus",
#                 branch="main",
#             )
#             self.fs.install_hooks()
#         except:
#             self.fs = list(DagsHubFilesystem.already_mounted_filesystems.values())[0]

#         self.img_paths = []
#         self.img_cache = {}  # path -> PIL
#         self.captions = []

#         with self.fs.open(self.dataset_root / "data/labels.tsv") as tsv:
#             tsv_lines = tsv.readlines()

#             rng = np.random.default_rng(seed)
#             if subset_size is not None:  # Random sample of subset
#                 indices = rng.choice(len(tsv_lines), size=subset_size, replace=False)
#                 indices = np.sort(indices)

#             for idx, row in enumerate(tsv_lines):
#                 if subset_size is not None and idx not in indices:
#                     continue
#                 row = row.strip()  # Rows are img_path, caption, score, url

#                 # Be careful here; rows are tab-separated, but some captions have tabs
#                 split_row = row.split("\t")
#                 img_path = split_row[0]
#                 caption = "\t".join(split_row[1:-2])

#                 self.img_paths.append(img_path)
#                 self.captions.append(caption)

#         self.metadata = pd.DataFrame(
#             {
#                 "image_path": self.img_paths,
#                 "caption": self.captions,
#             }
#         )

#     def __getitem__(self, idx):
#         img_path = self.img_paths[idx]
#         if img_path in self.img_cache:
#             img = self.img_cache[img_path]
#         else:
#             try:
#                 img = Image.open(self.dataset_root / "data" / self.img_paths[idx])
#             except (httpx.ReadTimeout, RuntimeError):
#                 # Just return a blank image
#                 if self.silent_timeout:
#                     img = Image.new("RGB", (512, 512))
#                 else:
#                     raise RuntimeError("Timeout upon image download from DagsHub")
#             img = img.convert("RGB")
#             self.img_cache[img_path] = img

#         if self.transform is not None:
#             img = self.transform(img)

#         caption = self.captions[idx]

#         return img, caption

#     def __len__(self):
#         return len(self.img_paths)

class LAIONMemorized(TorchDataset):
    """The Stable Diffusion-memorized samples of LAION surfaced by Webster (2023)."""

    # def __init__(
    #     self,
    #     transform=None,
    #     mem_type: Literal["MV", "RV", "TV", "N"] = "MV",
    #     return_on_timeout: Literal["blank", "null", "skip"] = "null",
    #     image_pkl="./outputs/laion_memorized.pkl",
    #     timeout=5,
    # ):
    def __init__(self, 
                 mem_type,
                 root_dir,
                 transform=None, 
                 return_on_timeout="skip",
                 ):
        """
        Args:
            transform: A transform to be applied to PILs when __getitem__ is called.
            mem_type: Memorized verbatim (MV), retrieval verbatim (RV), template verbatim (TV),
                or not memorized (N). See Webster (2023) for details.
            return_on_timeout: Behaviour for images that throw connection errors. "Blank" returns
                a black images, "null" returns None, and "skip" removes it from the dataset entirely.
            image_pkl: Location where cached images are stored (in .pkl format).
            timeout: Timeout period for loading images.
        """
        self.root_dir = Path(root_dir)

        self.transform = transform

        # Download the dataframe of memorized examples
        mem_url = "https://huggingface.co/datasets/fraisdufour/templates-verbs/resolve/main/groundtruth_parquets/sdv1_bb_edge_groundtruth.parquet"
        mem_df = pd.read_parquet(mem_url)
        print('debug mem_df:', mem_df.columns)

        # Create list of (img, caption) pairs
        mem_df = mem_df[mem_df["overfit_type"] == mem_type]  # Filter by mem_type
        print('debug mem_df filtered:', mem_df)
        
        # 251116 As requests.get to outside seems not allowed, this fuction was replaced with the following. 
        #url_to_pil = self._load_pils_from_urls_or_disk(mem_df, image_pkl)
        url_to_pil = self._load_pils_from_local(mem_df)

        
        self.imgs = []
        self.captions = []
        self.img_paths = []

        # for idx, row in mem_df.iterrows():
        #     if mem_type == "MV":
        #         img = url_to_pil[row["url"]]
        #         self._append_img_to_dataset(img, row["caption"], row["url"], return_on_timeout)
        #     elif mem_type in ("TV", "RV"):
        #         for img_url in row["retrieved_urls"]:
        #             img = url_to_pil[img_url]
        #             self._append_img_to_dataset(img, row["caption"], img_url, return_on_timeout)
        #     elif mem_type == "N":
        #         self._append_img_to_dataset(img, row["caption"], row["url"], return_on_timeout)
        
        for idx, row in mem_df.iterrows():
            if mem_type == "MV":
                img = url_to_pil[row["url"]]
                if img is not None:
                    self.imgs.append(img)
                    self.img_paths.append(row["url"])
                    self.captions.append(row["caption"])
            elif mem_type in ("TV", "RV"):
                for img_url in row["retrieved_urls"]:
                    img = url_to_pil[img_url]
                    if img is not None:
                        self.imgs.append(img)
                        self.img_paths.append(img_url)
                        self.captions.append(row["caption"])
            elif mem_type == "N":
                img = url_to_pil[row["url"]]
                if img is not None:
                    self.imgs.append(img)
                    self.img_paths.append(row["url"])
                    self.captions.append(row["caption"])
            
              # now actually a path string
        self.metadata = pd.DataFrame(
            {
                "image_path": self.img_paths,
                "caption": self.captions,
            }
        )
    def _local_path_for(self, overfit_type, idx, url, subidx=None):
        path = urlparse(url).path
        ext = os.path.splitext(path)[1] or ".jpg"
        ext = ext.lower()
        if subidx is None:
            fname = f"{idx}{ext}"
        else:
            fname = f"{idx}_{subidx}{ext}"
        return self.root_dir / overfit_type / fname

    def _load_pils_from_local(self, mem_df):
        url_to_pil = {}

        # (1) MV / N urls
        for idx, row in tqdm(mem_df.iterrows(), desc="Loading images"):
            t = row["overfit_type"]
            if t in ("MV", "N"):
                url = row["url"]
                path = self._local_path_for(t, idx, url)
                if path.exists():
                    try:
                        url_to_pil[url] = Image.open(path).convert("RGB")
                    except:
                        url_to_pil[url] = None
                        print(f"WARN: failed to read {path}")
   
                else:
                    #print(f"warn: {path} doesn't exist. maybe deleted for 0 byte size")
                    url_to_pil[url] = None

            # (2) RV / TV retrieved urls
            elif t in ("RV", "TV"):
                for k, url in enumerate(row["retrieved_urls"]):
                    path = self._local_path_for(t, idx, url, subidx=k)
                    if path.exists():
                        url_to_pil[url] = Image.open(path).convert("RGB")
                    else:
                        #print(f"warn: {path} doesn't exist. maybe deleted for 0 byte size")
                        url_to_pil[url] = None

        return url_to_pil

# 251116 As requests.get to outside seems not allowed, this fuction was replaced with the above. 
#
#     def _load_pils_from_urls_or_disk(self, mem_df, image_pkl):
#         """Compile the urls of all memorized images"""
#         # Check for PILs saved to a pickle on disk; otherwise, download them from online
#         image_pkl = Path(image_pkl)
#         print('debug image_pkl.exists():', image_pkl.exists())
#         #if image_pkl.exists():
#         if False:
#             with open(image_pkl, "rb") as f:
#                 print('debug 1')
#                 url_to_pil = pickle.load(f)
#             print('debug 2')

#         else:
#             image_pkl.parent.mkdir(exist_ok=True)

#             # All image urls (both matching and retrieved) in dataframe
#             all_memorized_urls = mem_df["url"].tolist() + list(
#                 chain.from_iterable(mem_df["retrieved_urls"])
#             )
#             all_memorized_urls = set(all_memorized_urls)

#             # Download the images into memory
#             url_to_pil = {}
#             timeouts = 0
#             for url in tqdm(all_memorized_urls, desc="Downloading images"):
#                 try:
#                     url_to_pil[url] = Image.open(
#                         requests.get(url, stream=True, timeout=timeout).raw
#                     )
#                 except:
#                     timeouts += 1
#                     url_to_pil[url] = None
#             if timeouts > 0:
#                 print(f"{timeouts} images not downloaded due to timeout")

#             # Save to disk
#             with open(image_pkl, "wb") as f:
#                 pickle.dump(url_to_pil, f)

#         #print('debug url_to_pil:', url_to_pil)
#         return url_to_pil

    def _append_img_to_dataset(self, img, caption, url, return_on_timeout):
        """Append image to dataset whilst handling missing image behaviour"""
        print('debug', img)
        if img is None and return_on_timeout == "skip":
            return

        if img is not None:
            self.imgs.append(img)
        elif return_on_timeout == "blank":
            self.imgs.append(None)
        elif return_on_timeout == "null":
            self.imgs.append(Image.new("RGB", (512, 512)))

        self.captions.append(caption)
        self.img_paths.append(url)

    def __getitem__(self, idx):
        if self.transform is not None:
            try:
                img = self.transform(self.imgs[idx])
            except:
                import ipdb

                ipdb.set_trace()
        else:
            img = self.imgs[idx]
        caption = self.captions[idx]
        return img, caption

    def __len__(self):
        return len(self.imgs)
