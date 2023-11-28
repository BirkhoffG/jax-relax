# Cell
# https://github.com/fastai/fastai/blob/master/fastai/imports.py
from __future__ import annotations
import matplotlib.pyplot as plt,numpy as np,pandas as pd,scipy
from typing import Union,Optional,Dict,List,Tuple,Sequence,Mapping,Callable,Iterable,Any,NamedTuple
import io,operator,sys,os,re,mimetypes,csv,itertools,json,shutil,glob,pickle,tarfile,collections
import hashlib,itertools,types,inspect,functools,time,math,bz2,typing,numbers,string
import multiprocessing,threading,urllib,tempfile,concurrent.futures,matplotlib,warnings,zipfile

# import sklearn
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate

# misc.
from pprint import pprint
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
from fastcore.utils import in_jupyter
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pydantic import BaseModel as BaseParser, validator, ValidationError, Field
import functools as ft
from functools import partial

# jax related
import jax
from jax import pmap, vmap, random, device_put, lax, jit, Array
import jax.numpy as jnp, jax.random as jrand
from jax_tqdm import loop_tqdm, scan_tqdm
import optax
import chex

# keras
os.environ['KERAS_BACKEND'] = 'jax'
# import keras_core as keras
import keras

# nn related
import haiku as hk
