from metrics import compute_clean_FID
from globals import DATA_ROOT

REAL_DIR = DATA_ROOT + "/true"
FAKE_DIR = DATA_ROOT + "/fake"
CONVERT_IMAGES = True

if __name__ == "__main__":

    compute_clean_FID(REAL_DIR, FAKE_DIR, CONVERT_IMAGES)