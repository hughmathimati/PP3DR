from datasets.rtmv.rtmv_dataset import rtmv_dataset

class rtmv_test_dataset(rtmv_dataset):
    def __init__(self):
        super().__init__(
            cache_path="/vulcanscratch/hughma/PP3DR/datasets/rtmv/rtmv_test_dataset_cache.pth",
            dir="/fs/vulcan-datasets/RTMV/40_scenes"
        )

import traceback
def helper(dataset, i):
    try:
        # Suppress standard output during the getitem to hide the massive OpenCV C++ spam
        _ = dataset[i]
        return i, True, None
    except Exception as e:
        # Capture the exact traceback!
        return i, False, traceback.format_exc()

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    dataset = rtmv_test_dataset()

    with ThreadPoolExecutor() as executor:
        future_to_idx = {executor.submit(helper, dataset, i): i for i in range(len(dataset))}

        for future in tqdm(as_completed(future_to_idx)):
            index, valid, error_trace = future.result()
            if not valid:
                print(f"\n{'=' * 60}")
                print(f"CRASH ON SEQUENCE {index}:")
                print(f"{'=' * 60}")
                print(error_trace)
                print(f"{'=' * 60}\n")
                print("Halting scan to fix the bug...")
                import sys

                sys.exit(1)

    print("\nAll sequences passed successfully!")