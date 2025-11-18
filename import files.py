import os
import shutil
import random
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
import kagglehub
import pandas as pd
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm 
import zipfile
import zipfile
import os
from pathlib import Path
import shutil
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt


!pip install ultralytics
